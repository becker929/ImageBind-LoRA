import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data


class Image2AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        device: str = "cpu",
    ):
        self.transform = transform
        self.device = device
        path_pairs = dict()
        for filename in os.listdir(data_dir):
            name = filename.split(".")[0].lower()
            if name not in path_pairs:
                path_pairs[name] = [None, None]
            if filename.endswith(".jpg"):
                path_pairs[name][0] = os.path.join(data_dir, filename)
            elif filename.endswith('.wav'):
                path_pairs[name][1] = os.path.join(data_dir, filename)
            else:
                raise ValueError(f"Invalid file extension {os.path.join(data_dir, filename)}")

        self.path_pairs = list(path_pairs.values())

        # check that every image has an audio file and vice versa
        for pair in self.path_pairs:
            if pair[0] is None or pair[1] is None:
                raise ValueError(f"Image and audio file must have the same name in {data_dir}")

        print(f"Loaded {len(self.path_pairs)} pairs of images and audio files from {data_dir}")

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, index):
        img_path, audio_path = self.path_pairs[0]
        images = data.load_and_transform_vision_data(
            [img_path], self.device, to_tensor=False
        )
        sounds = data.load_and_transform_audio_data([audio_path], self.device)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        return images, ModalityType.VISION, sounds, ModalityType.AUDIO
