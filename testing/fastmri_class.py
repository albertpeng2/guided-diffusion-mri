import h5py
import random
import torch
from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas
import os

DATASHEET_PATH = '/home/research/a.peng/NeuralCompression/data/'

DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_multicoil.csv'))
def INDEX2_helper(idx, key_):
    file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]
    #print(file_id_df)

    assert len(file_id_df.index) == 1
    #print(file_id_df[idx])
    return file_id_df[idx]


INDEX2FILE = lambda idx: INDEX2_helper(idx, 'FILE')


def INDEX2DROP(idx):
    ret = INDEX2_helper(idx, 'DROP')

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True



class FastMRI(torch.utils.data.Dataset):

    SINGLECOIL_MIN_VAL = 4.4726e-09
    SINGLECOIL_MAX_VAL = 0.0027
    MULTICOIL_MIN_VAL = 1.0703868156269891e-06
    MULTICOIL_MAX_VAL = 0.0007881390047259629

    def __init__(
        self,
        root: Union[str, Path],
        challenge: str,
        split: str,
        normalize: bool = True,
        machine_type: str = "AXT2",
        num_slices: int = 16,
        patch_shape: Union[int, List[int]] = -1,
        transform: Optional[Callable] = None,
    ):
        """
        Dataset of 3D MRI scans.

        Args:
            root (Union[str, Path]): Path to the dataset root
            challenge (str): "singlecoil" or "multicoil"
            split (str, optional): "train", "val", or "test"
            normalize (bool, optional): Whether to normalize data to lie in [0, 1].
                Defaults to True.
            machine_type (str): If not None, machine type to use. Otherwise uses all
                machine types.
            num_slices (int): If not None, filter for volumes with the specified number
                of slices.
            patch_shape (Union[int, List[int]], optional): If not -1, perform random
                crops of shape patch_shape. Defaults to -1.
            transform (Optional[Callable], optional): [description]. Defaults to None.
        """
        if challenge not in ["singlecoil", "multicoil"] or split not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError

        root = Path(root)
        #print(f"root is : {root}")
        files = sorted(list(Path(root).glob("*.h5")))
        #print(type(files))
        #print(type(files[0]))
        #print("done sorting")

        filesNames = []

        startIdx = 563
        endIdx = 1377

        if split == "train":
            endIdx = 1289
        else: #test
            startIdx = 1290

        for idx in range(startIdx, endIdx):
            if not INDEX2DROP(idx):
                file_id_df = DATASHEET['FILE'][DATASHEET['INDEX'] == idx]
                full_path = "/project/cigserver4/export2/Dataset/fastmri_brain_multicoil/" + str(file_id_df[idx]) + ".h5"
                full_path = Path(full_path)
                filesNames.append(full_path)
                #print(full_path)

        files = filesNames


        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        if not root.exists or len(files) == 0:
            raise FileNotFoundError

        self.normalize = normalize
        self.machine_type = machine_type
        self.num_slices = num_slices
        self.patch_shape = patch_shape

        # self.random_crop = patch_shape != -1  // source
        self.random_crop = 0
        self.transform = transform

        if challenge == "singlecoil":
            self.min_val = FastMRI.SINGLECOIL_MIN_VAL
            self.max_val = FastMRI.SINGLECOIL_MAX_VAL
        else:
            self.min_val = FastMRI.MULTICOIL_MIN_VAL
            self.max_val = FastMRI.MULTICOIL_MAX_VAL

        # Optionally filter for machine type
        if machine_type is not None:
            valid_files = []
            for path in files:
                if get_machine_type(path) == machine_type:
                    valid_files.append(path)
            self.files = valid_files
        else:
            self.files = files

        # Optionally filter for volumes with a specific number of slices
        if num_slices is not None:
            self.files = list(filter(self.num_slices_equal_to, self.files))


    def num_slices_equal_to(self, file):
        with h5py.File(file, "r") as f:
            mri = torch.from_numpy(f[self.recons_key][()])
            return mri.shape[0] == self.num_slices

    def __getitem__(self, idx):

        with h5py.File(self.files[idx], "r") as f:
            mri = torch.from_numpy(f[self.recons_key][()])
        print(f"mri size is {mri.size()}")

        # Shape ({1,} depth, height, width)
        if mri.ndim == 3:
            # Ensure volume has a channel dimension, i.e. (1, depth, height, width)
            mri = mri.unsqueeze(0)

        # Normalize data to lie in [0, 1]的
        if self.normalize:
            mri = (mri - self.min_val) / (self.max_val - self.min_val)
            mri = torch.clamp(mri, 0.0, 1.0)

        if self.transform:
            mri = self.transform(mri)

        #print(f"shape of mri before :{mri.shape}")
        if self.random_crop:
            #print(mri.shape)
            mri = random_crop3d(mri, self.patch_shape)

        #print(f"shape of mri after is :{mri.shape}")

        return mri

    def __len__(self):
        return len(self.files)

# def random_crop3d(data, patch_shape):
#     #patch_shape = [10, 260, 260]
#     print(f"data shape:{data.shape}")
#     print(f"patch shape:{patch_shape.shape}")
#
#     depth_from = 0
#     height_from = 0
#     width_from = 0
#     return data[
#         ...,
#         depth_from: depth_from + patch_shape[0],
#         height_from: height_from + patch_shape[1],
#         width_from: width_from + patch_shape[2],
#     ]

def random_crop3d(data, patch_shape):
    #print(f"data shape:{data.shape}")
    #print(f"patch size is  {len(patch_shape)} and patch is {patch_shape}")
    if not (
        0 < patch_shape[0] <= data.shape[-3]
        and 0 < patch_shape[1] <= data.shape[-2]
        and 0 < patch_shape[2] <= data.shape[-1]
    ):
        #print(data.shape)
        #print(patch_shape)
        raise ValueError("Invalid shapes.")
    depth_from = random.randint(0, data.shape[-3] - patch_shape[0])
    height_from = random.randint(0, data.shape[-2] - patch_shape[1])
    width_from = random.randint(0, data.shape[-1] - patch_shape[2])
    return data[
        ...,
        depth_from : depth_from + patch_shape[0],
        height_from : height_from + patch_shape[1],
        width_from : width_from + patch_shape[2],
    ]


def get_machine_type(path):
    """Returns machine type from path to fastMRI file."""
    # Get filename
    filename = str(path).split("/")[-1]
    # Remove 'file_brain_' string which is at the beginning of every filename
    # Then extract machine type (first word after file_brain_)
    return filename.replace("file_brain_", "").split("_")[0]
