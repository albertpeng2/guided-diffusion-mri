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


def load_data(image_size):
    my_dataset = MRIDataset(image_size)
    return my_dataset


class MRIDataset(Dataset):

    def __init__(
            self,
            image_size
    ):
        super().__init__()
        self.image_size = image_size

        kspace_imgs = []

        for idx in range(1000, 1350):
            if INDEX2DROP(idx):  # if we need to drop this
                continue
            filename = INDEX2FILE(idx) + ".h5"
            file_full_path = os.path.join(MRI_PATH, filename)

            # print(file_full_path)
            with h5py.File(file_full_path, 'r') as f:
                y = f['kspace'][:]
                # y would be something like (16, 20, 768, 396) for (slice, coil, height, width)

                slice_kspace = y[0]  # TODO: change this?
                kspace_imgs.append(slice_kspace)

        self.kspace_imgs = kspace_imgs

    def __len__(self):
        return len(self.kspace_imgs)

    def __getitem__(self, idx):
        kspace_img = self.kspace_imgs[idx]

        print(f"raw kspace_image has size {kspace_img.shape}")

        kspace_img = T.to_tensor(kspace_img)

        slice_img = fastmri.ifft2c(kspace_img)
        slice_img2 = fastmri.complex_abs(slice_img)
        slice_img3 = fastmri.rss(slice_img2, dim=0)

        target_size = torch.ones(396, 396)

        torch.set_printoptions(threshold=1000000)
        img = torch.nn.functional.normalize(slice_img3, 2, 0)
        img = torch.nn.functional.normalize(img, 2, 1)

        print(img.shape)
        img = center_crop_arr(img, target_size.shape)

        save_image(img, "normal_img.png")

        img /= torch.max(img)
        print(img)
        save_image(img, "new_img.png")
        save_image(img, "new_img.jpeg")

        mask_func = subsample.EquispacedMaskFractionFunc([0.05], [4], seed=5)
        masked_kspace, mask, _ = T.apply_mask(kspace_img, mask_func)
        sampled_image = fastmri.ifft2c(masked_kspace)  # Apply Inverse Fourier Transform to get the complex image
        sampled_image_abs = fastmri.complex_abs(sampled_image)  # Compute absolute value to get a real image
        sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

        sampled_image = torch.nn.functional.normalize(sampled_image_rss, 2, 0)
        sampled_image = torch.nn.functional.normalize(sampled_image, 2, 1)
        print(f"sampled image shape: {sampled_image.shape}")
        # sampled_image_crop = center_crop_arr(sampled_image, target_size.shape)
        # sampled_image_crop /= torch.max(sampled_image_crop)
        # save_image(sampled_image_crop, "sampled_img.png")

        sampled_image /= torch.max(sampled_image)
        save_image(sampled_image, "sampled_img.png")

        print(img.shape)
        return img


def center_crop_arr(img, target_shape):
    # crops image_size from pil_image.size, only for 2d images
    img_shape = img.shape
    print(f"input size in center crop: {img.shape}")
    assert target_shape[0] <= img_shape[0]
    assert target_shape[1] <= img_shape[1]

    margin_y = (img_shape[0] - target_shape[0]) // 2
    margin_x = (img_shape[1] - target_shape[1]) // 2
    ret = img[margin_y:img_shape[0] - margin_y, margin_x:img_shape[1] - margin_x]
    print(ret.shape)
    return ret


dataset = load_data(np.array([2, 3]))
loader = DataLoader(dataset, shuffle=True)

for im in loader:
    print(im.shape)
