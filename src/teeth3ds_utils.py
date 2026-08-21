def load_official_splits(config, root_path) -> tuple[list[str], list[str]]:
    """
    Loads the official "3D Teeth Seg Challenge" train/test split:
        - train split: publicly available for training
        - test split: private test split (during challenge) for evaluation
    """
    base_path = (
        root_path
        / config["dataset_path_3d"]
        / "raw"
        / "3DTeethSeg22_challenge_train_test_split"
    )

    train_files = ["public-training-set-1.txt", "public-training-set-2.txt"]
    test_files = ["private-testing-set.txt"]

    def load_files(filenames: list[str]) -> list[str]:
        data = []
        for name in filenames:
            file = base_path / name
            content = file.read_text(encoding="utf-8").splitlines()
            data.extend([line for line in content if line.strip()])
        data.sort()
        return data

    train_split = load_files(train_files)
    test_split = load_files(test_files)

    return train_split, test_split
