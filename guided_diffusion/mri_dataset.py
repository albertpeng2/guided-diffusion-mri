import pandas
import os
import time
import h5py
import fastmri
import torch.nn
from fastmri.data import transforms as T
from fastmri.data import subsample
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import numpy as np

MRI_PATH = "/project/cigserver4/export2/Dataset/fastmri_brain_multicoil/"
DATASHEET_PATH = '/project/cigserver1/export1/a.peng/guided-diffusion-mri'
DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_multicoil.csv'))


def INDEX2_helper(idx, key_):
    file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]

    assert len(file_id_df.index) == 1
    return file_id_df[idx]


def INDEX2FILE(idx):
    return INDEX2_helper(idx, "FILE")


def INDEX2DROP(idx):
    ret = INDEX2_helper(idx, 'DROP')

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True


def load_data(image_size, batch_size, shuffle = False):
    mri_dataset = MRIDataset(image_size)
    mri_loader = DataLoader(
        mri_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    return mri_loader


class MRIDataset(Dataset):

    def __init__(
            self,
            image_size
    ):
        super().__init__()

        self.image_size = image_size

        kspace_slices = []

        for idx in range(1000, 1350):
            if INDEX2DROP(idx):  # if we need to drop this
                continue
            filename = INDEX2FILE(idx) + ".h5"
            file_full_path = os.path.join(MRI_PATH, filename)

            # print(file_full_path)
            with h5py.File(file_full_path, 'r') as f:
                y = f['kspace'][:]
                # y would be something like (16, 20, 768, 396) for (slice, coil, height, width)
                num_slices = y.shape[0]
                edge = 3
                # ignore first 3 and last 3 slices
                for i in range(num_slices-3):
                    slice_kspace = y[i+edge]
                    kspace_slices.append(slice_kspace)

        self.kspace_imgs = kspace_slices

    def __len__(self):
        return len(self.kspace_imgs)

    def __getitem__(self, idx):
        kspace_img = self.kspace_imgs[idx]

        #print(f"raw kspace_image has size {kspace_img.shape}")

        kspace_img = T.to_tensor(kspace_img)
        slice_img = fastmri.ifft2c(kspace_img)
        slice_img2 = fastmri.complex_abs(slice_img)
        slice_img3 = fastmri.rss(slice_img2, dim=0)

        #crop, normalize and save normal image
        target_size = torch.ones(396, 396)
        img = center_crop_arr(slice_img3, target_size.shape)
        img = (img - torch.min(img))/torch.max(img)
        img = torch.unsqueeze(img, 0)
        #print(f"img shape: {img.shape}")
        # save_image(img, "new_img.png")

        # mask kspace
        mask_func = subsample.EquispacedMaskFractionFunc([0.05], [4], seed=5)
        masked_kspace, mask, _ = T.apply_mask(kspace_img, mask_func)
        sampled_image = fastmri.ifft2c(masked_kspace)  # Apply Inverse Fourier Transform to get the complex image
        sampled_image_abs = fastmri.complex_abs(sampled_image)  # Compute absolute value to get a real image
        sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

        #normalzie and save masked image
        sampled_image_crop = center_crop_arr(sampled_image_rss, target_size.shape)
        norm_sampled_img = (sampled_image_crop - torch.min(sampled_image_crop)) / torch.max(sampled_image_crop)
        norm_sampled_img = torch.unsqueeze(norm_sampled_img, 0)
        #print(f"img shape: {norm_sampled_img.shape}")
        # save_image(norm_sampled_img, "sampled_img.png")

        out_dict = {"masked_img": np.array(norm_sampled_img, dtype=np.int64)}

        return img, out_dict

def center_crop_arr(img, target_shape):
    # crops image_size from pil_image.size, only for 2d images
    img_shape = img.shape
    #print(f"input size in center crop: {img.shape}")
    assert target_shape[0] <= img_shape[0]
    assert target_shape[1] <= img_shape[1]

    margin_y = (img_shape[0] - target_shape[0]) // 2
    margin_x = (img_shape[1] - target_shape[1]) // 2
    ret = img[margin_y:img_shape[0] - margin_y, margin_x:img_shape[1] - margin_x]
    #print(ret.shape)
    return ret

#
# torch.set_printoptions(threshold=1000000)
#
# dataset = load_data(np.array([2, 3]))
# loader = DataLoader(dataset, shuffle=True)
# #print(f"length: {len(loader)}")
