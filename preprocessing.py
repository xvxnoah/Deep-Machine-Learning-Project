import numpy as np
import pydicom
import matplotlib.pyplot as plt
import math


def convert_file_to_3channel(dcm_path):
    # Loading the file data
    ds = pydicom.dcmread(dcm_path)
    # Loading the pixel values
    img = ds.pixel_array.astype(np.float32)
    # Converting to Hounsfield Units (HU)
    intercept = float(ds.RescaleIntercept)
    slope = float(ds.RescaleSlope)
    hu = img * slope + intercept
    # Defining the window presets
    # brain, subdural and bone
    windows = [(40, 80), (80, 200), (600, 2800)]
    # Windowing the images according to the three presets
    channels = []
    for center, width in windows:
        mn = center - width // 2
        mx = center + width // 2
        win = np.clip(hu, mn, mx)
        # Normalizing to [0, 1]
        norm = (win - mn) / (mx - mn)
        # Resizing to the output size
        channels.append(norm)
    # Stacking into (3, H, W)
    three_chan = np.stack(channels, axis=0).astype(np.float32)
    return three_chan


def display_triplets(triplets_data, triplets_per_row=4):
    # Calculating the number of rows and columns needed
    num_triplets = len(triplets_data)
    num_rows = math.ceil(num_triplets / triplets_per_row)
    # Plotting for each row
    for row_number in range(num_rows):
        row_triplets_data = triplets_data[
            row_number * triplets_per_row : (row_number + 1) * triplets_per_row
        ]
        fig, axes = plt.subplots(
            3, triplets_per_row * 3, figsize=(triplets_per_row * 7.5, 7.5)
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # Sweeping over all the row triplet data
        for i, triplet in enumerate(row_triplets_data):
            triplet_slices = triplet["slices"]
            triplet_label = triplet["label"]
            # Calculating the triplet's position in the grid
            triplet_col = i % triplets_per_row
            # Looping over each triplet
            for slice_idx in range(3):
                dcm_path = triplet_slices[slice_idx]
                # Processing the DICOM file to get 3 channels
                three_chan_img = convert_file_to_3channel(dcm_path)
                # Viewing the three channels per slice
                for channel_idx in range(3):
                    # Viewing the data
                    ax = axes[slice_idx, triplet_col * 3 + channel_idx]
                    ax.imshow(three_chan_img[channel_idx, :, :])
                    if slice_idx == 0 and channel_idx == 1:
                        ax.set_title(
                            f"|--------------Label: {triplet_label}--------------|",
                            fontsize=20,
                            pad=10,
                        )
                    ax.axis("off")
        plt.show()
