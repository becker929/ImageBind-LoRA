import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data


class Tuning4Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        device: str = "cpu",
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.path_pairs = {}
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg") or filename.endswith(".wav"):
                name = filename.split(".")[0]
                if name not in self.path_pairs:
                    self.path_pairs[name] = [None, None]
                if filename.endswith(".jpg"):
                    self.path_pairs[name][0] = os.path.join(root_dir, filename)
                else:  # filename.endswith('.wav`````````````````````````````````````````````````````````````')
                    self.path_pairs[name][1] = os.path.join(root_dir, filename)

        self.path_pairs = list(self.path_pairs.values())

        # Split dataset
        train_pairs = self.path_pairs[:3]
        test_pairs = self.path_pairs[3:]

        if split == "train":
            self.path_pairs = train_pairs
        elif split == "test":
            self.path_pairs = test_pairs
        else:
            raise ValueError(
                f"Invalid split argument. Expected 'train' or 'test', got {split}"
            )

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, index):
        print("getting item: ", index)
        img_path, audio_path = self.path_pairs[0]
        images = data.load_and_transform_vision_data(
            [img_path], self.device, to_tensor=False
        )
        sounds = data.load_and_transform_audio_data([audio_path], self.device)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        return images, ModalityType.VISION, sounds, ModalityType.AUDIO
