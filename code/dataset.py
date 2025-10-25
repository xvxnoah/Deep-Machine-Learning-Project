import json
import os
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from preprocessing import convert_file_to_3channel
import torch


class PreprocessedDataset(Dataset):
    """Dataset for loading preprocessed .pt files."""

    def __init__(self, root_dir, tranform=None):
        """
        Args:
            root_dir (string): Directory with all the preprocessed files,
                               e.g., 'processed_data/train'.
        """
        self.root_dir = root_dir
        self.transform = tranform
        # Recursively find all files ending with .pt in the directory
        self.file_paths = sorted(
            glob(os.path.join(root_dir, "**", "*.pt"), recursive=True)
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads a tensor and its label from a .pt file.
        """
        file_path = self.file_paths[idx]

        # Load the saved data (expected to be a tuple)
        # weights_only=False is safe here since we control the preprocessed data
        image_tensor, label = torch.load(file_path, weights_only=False)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Convert label to tensor if it's a list
        if isinstance(label, list):
            label = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label


class DicomSeriesDataset(Dataset):
    """Dataset for loading DICOM series."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the patient subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Get a list of all subdirectories for each group containing tiplet folders
        self.triplet_folders = [
            f.path
            for group in os.scandir(root_dir)
            if group.is_dir()
            for f in os.scandir(group)
            if f.is_dir()
        ]
        self.labels = json.load(open(os.path.join(root_dir, "labels.json")))

    def __len__(self):
        return len(self.triplet_folders)

    def __getitem__(self, idx):
        triplet_folder = self.triplet_folders[idx]
        triplet_id = os.path.basename(triplet_folder)

        # Find all .dcm files in the folder
        dicom_files = sorted(glob(os.path.join(triplet_folder, "*.dcm")))

        # Read the DICOM files, apply 3-channel windowing and stack their pixel arrays
        images = []
        for file_path in dicom_files:
            channels = convert_file_to_3channel(file_path)
            images.append(channels)

        # Stack the images into a single tensor of shape (9, H, W)
        image_stack = np.concatenate(images)

        assert image_stack.shape[0] == len(dicom_files) * 3, (
            f"Expected 9 images, but got {image_stack.shape[0]}"
        )

        if self.transform:
            image_stack = self.transform(image_stack)

        labels = self.labels[triplet_id]

        return image_stack, labels

if __name__ == "__main__":
    # Example usage

    from torch.utils.data import DataLoader

    dataset = DicomSeriesDataset(root_dir="triplets")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        # Just process a few batches for demonstration
        if i == 2:
            break

