from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    camera_position_from_spherical_angles,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


class Preprocessing:
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

        cameras = FoVPerspectiveCameras(
            znear=0.1,
            zfar=100.0,
            fov=self.config["2d_projection"]["fov"],
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

        blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

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
        images = self.render_2d_images(mesh)
        masks = self.render_2d_masks(mesh, vertex_labels)
        return images, masks

    def render_2d_images(self, mesh: Meshes) -> NDArray[np.uint8]:
        # define a default color for each vertex
        num_vertices = mesh.verts_packed().shape[0]
        verts_features = (
            torch.ones((1, num_vertices, 3), dtype=torch.float32, device=self.device)
            * 0.75
        )
        mesh.textures = TexturesVertex(verts_features=verts_features)

        meshes = mesh.extend(self.views.shape[0])
        images = self.image_renderer(meshes)

        # convert float images to grayscale, scale, and convert to uint8
        gray_images = images[..., :3].mean(dim=-1)
        uint8_images = (gray_images * 255.0).clamp(0, 255).to(torch.uint8)

        # bring the entire batch to CPU and convert to NumPy
        return uint8_images.cpu().numpy()

    def render_2d_masks(self, mesh: Meshes, vertex_labels: NDArray[np.uint8]):
        # render 2D projection label masks (face index rasterization)
        # Instead of rendering colors and guessing pixels, this method uses PyTorch3D’s rasterizer to determine exactly
        # which face index is visible at every pixel. It then maps that face back to its vertex labels.

        meshes = mesh.extend(self.views.shape[0])

        # get fragments (pix_to_face contains the face index for each pixel)
        fragments = self.mask_rasterizer(meshes)
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
