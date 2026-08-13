from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    camera_position_from_spherical_angles,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


class ViewProjector:
    def __init__(
        self, config: dict[str, Any], device: str | torch.device | None = None
    ) -> None:
        self.config = config
        self.device = device

        self.background_value = config["2d_projection"]["background_value"]
        # each row in views is [elevation, azimuth]
        self.views = np.array(config["2d_projection"]["views"])

        R, T = look_at_view_transform(
            dist=self.config["2d_projection"]["distance"],
            elev=self.views[:, 0],
            azim=self.views[:, 1],
            device=self.device,
        )

        # calc orthographic bounds (from perspective fov and dist)
        # Formula: 80 * tan(60 / 2 * pi / 180) = 46.188
        ortho_size = self.config["2d_projection"]["distance"] * np.tan(
            self.config["2d_projection"]["fov"] / 2.0 * np.pi / 180.0
        )

        cameras = FoVOrthographicCameras(
            znear=0.1,
            zfar=100.0,
            max_y=ortho_size,   
            min_y=-ortho_size,
            max_x=ortho_size,   
            min_x=-ortho_size,
            R=R,
            T=T,
            device=self.device,
        )

        raster_settings = RasterizationSettings(
            image_size=self.config["2d_projection"]["image_size"],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.image_renderer = self.__init_image_renderer(cameras, raster_settings)
        self.mask_rasterizer = self.__init_mask_rasterizer(cameras, raster_settings)

    def __init_image_renderer(self, cameras, raster_settings) -> MeshRenderer:
        light_dir = camera_position_from_spherical_angles(
            distance=1.0,
            elevation=self.views[:, 0],
            azimuth=self.views[:, 1],
            device=self.device,
        )

        lights = DirectionalLights(
            direction=light_dir,
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),),
            specular_color=(
                (0.2, 0.2, 0.2),
            ),  # reduced specular highlight to keep tooth boundaries matte
            device=self.device,
        )

        bg_rgb_val = self.background_value / 255.0
        blend_params = BlendParams(background_color=(bg_rgb_val, bg_rgb_val, bg_rgb_val))

        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                cameras=cameras,
                lights=lights,
                blend_params=blend_params,
                device=self.device,
            ),
        )

    def __init_mask_rasterizer(self, cameras, raster_settings) -> MeshRasterizer:
        return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    def render_2d_views(
        self, mesh: Meshes, vertex_labels: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        images = self.render_2d_images_np(mesh)
        masks = self.render_2d_masks(mesh, vertex_labels)
        return images, masks

    def render_2d_images(self, mesh: Meshes) -> torch.Tensor:
        # define a default color for each vertex
        num_vertices = mesh.verts_packed().shape[0]
        verts_features = (
            torch.ones((1, num_vertices, 3), dtype=torch.float32, device=self.device)
            * 0.75
        )
        mesh.textures = TexturesVertex(verts_features=verts_features)

        meshes_ext = mesh.extend(self.views.shape[0])
        return self.image_renderer(meshes_ext)

    def render_2d_images_np(self, mesh: Meshes) -> NDArray[np.uint8]:
        images = self.render_2d_images(mesh)

        # convert float images to grayscale, scale, and convert to uint8
        gray_images = images[..., :3].mean(dim=-1)
        uint8_images = (gray_images * 255.0).clamp(0, 255).to(torch.uint8)

        # bring the entire batch to CPU and convert to NumPy
        return uint8_images.cpu().numpy()

    def render_2d_masks(
        self, mesh: Meshes, vertex_labels: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        # render 2D projection label masks (face index rasterization)
        # Instead of rendering colors and guessing pixels, this method uses PyTorch3D’s rasterizer to determine exactly
        # which face index is visible at every pixel. It then maps that face back to its vertex labels.

        meshes_ext = mesh.extend(self.views.shape[0])

        # get fragments (pix_to_face contains the face index for each pixel)
        fragments = self.mask_rasterizer(meshes_ext)
        # Shape: (N, H, W) -> values are face indices, -1 means background
        pix_to_face = fragments.pix_to_face[..., 0]

        # map faces to vertex labels (identical for all view since it's the same mesh)
        # faces shape: (F, 3) | vertex_labels_tensor shape: (V,)
        faces = mesh.faces_packed()
        vertex_labels_tensor = torch.from_numpy(vertex_labels).to(mesh.device)

        # get the labels of the 3 vertices for every face -> Shape: (F, 3)
        face_vert_labels = vertex_labels_tensor[faces]

        # TODO, which method?
        # define face label by taking the first vertex
        face_labels = face_vert_labels[:, 0]  # Shape: (F,)
        # define face label by majority vote (or taking the first vertex)
        # face_labels = torch.mode(face_vert_labels, dim=1).values # Shape: (F,)
        face_labels = face_labels.repeat(self.views.shape[0])

        # add a background label
        face_labels_with_bg = torch.cat(
            [face_labels, torch.tensor([self.background_value], device=mesh.device)]
        )

        # generate the final 2D segmentation mask
        return face_labels_with_bg[pix_to_face].cpu().numpy().astype(np.uint8)

    def back_project_vertex_labels(
        self, mesh: Meshes, masks: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Project 2D mask labels back to 3D mesh vertices using face indexing.

        masks: Shape (N_views, H, W)
        """
        # num_classes must be the max label value + 1, not the count of unique labels
        # this ensures the voting matrix is wide enough so that every label value can be used as a direct column index
        num_classes = int(masks.max()) + 1

        # get face information
        # when extending a mesh in PyTorch3D, face indices in fragments.pix_to_face
        # refer to the indices in the original mesh's faces_packed() list
        faces = mesh.faces_packed()  # (F, 3)
        num_faces_per_mesh = faces.shape[0]
        num_verts = mesh.verts_packed().shape[0]
        num_views = self.views.shape[0]

        # re-run rasterizer to get pix_to_face
        # this tells us which face index is at every pixel (u, v)
        meshes_ext = mesh.extend(num_views)
        fragments = self.mask_rasterizer(meshes_ext)
        pix_to_face = fragments.pix_to_face[..., 0]  # (N_views, H, W)

        # prepare tensors
        masks_tensor = torch.from_numpy(masks).to(self.device).long()

        # filter background
        # create a mask of pixels that actually hit the mesh (PyTorch3D returns -1 for background in pix_to_face)
        hit_mask = pix_to_face >= 0

        # extract valid data
        # use modulo operator to map global batch face index to local mesh face index
        valid_face_indices = pix_to_face[hit_mask] % num_faces_per_mesh
        valid_labels = masks_tensor[hit_mask]

        # TODO: exclude background from voting
        # valid_mask = (valid_labels != self.background_value)
        # valid_face_indices = valid_face_indices[valid_mask]
        # valid_labels = valid_labels[valid_mask]

        # map pixel labels to vertices
        # a pixel belongs to a face, a face has 3 vertices
        # map the pixel's label as a vote to all 3 vertices of that face
        face_verts = faces[valid_face_indices]

        # scatter the votes into the (V, C) buffer
        # one pixel votes for 3 vertices, repeat labels 3 times
        v_indices = face_verts.reshape(-1)  # flattened vertex indices
        l_indices = valid_labels.repeat_interleave(3)  # flattened labels

        if not (l_indices < num_classes).all():
            raise ValueError(
                f"Label found in mask ({l_indices.max()}) exceeds num_classes ({num_classes})"
            )

        # voting buffer (V, num_classes)
        votes = torch.zeros((num_verts, num_classes), device=self.device)
        # Use a dummy tensor of ones to count occurrences
        ones = torch.ones_like(v_indices, dtype=torch.float32)

        # advanced indexing to populate the votes
        # votes[v_indices, l_indices] += 1
        # Note: simple += doesn't work for duplicate indices in PyTorch, we must use put_ or scatter_add_

        # create a flat index for the (V, C) matrix to use scatter_add_
        flat_indices = v_indices * num_classes + l_indices
        votes_flat = votes.view(-1)
        votes_flat.scatter_add_(0, flat_indices, ones)

        # consensus
        # TODO optionally:
        # Ignore the background class (e.g. index 0 or self.background_value) during the argmax
        # so vertices aren't labeled as background just because they were invisible in some views.
        # votes[:, self.background_value] = 0

        # TODO, distinguish background and vertices without votes, new class id "unknown"?
        vertex_labels = torch.argmax(votes, dim=1)
        return vertex_labels.cpu().numpy().astype(np.uint8)
