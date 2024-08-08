import h5py
import os
import numpy as np
import tqdm
import torch
import tifffile as tiff
import pandas
from torch.utils.data import Dataset
import pygrappa.grappa as grappa
from typing import Tuple
import torch.nn.functional as F

#old limited brain dataset in server4
ROOT_PATH = '/project/cigserver4/export2/Dataset/fastmri_brain_multicoil'
DATASHEET_PATH = '/project/cigserver4/export3/a.peng/I2SB/dataset'
DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_multicoil.csv'))

# #limited brain dataset in server5
# ROOT_PATH = '/project/cigserver4/export2/Dataset/fastmri_brain_multicoil'
# # ROOT_PATH = '/project/cigserver5/export1/a.peng/Dataset/fastmri/data/brain/multicoil_train'
# DATASHEET_PATH = '/project/cigserver4/export3/a.peng/I2SB/dataset'
# DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_multicoil.csv'))

#full brain dataset
# ROOT_PATH = '/project/cigserver5/export1/a.peng/Dataset/fastmri/data/brain/multicoil_train'
# DATASHEET_PATH = '/project/cigserver5/export1/a.peng/Dataset/fastmri/data/brain/'
# DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_train.csv'))


def grappa_image(y, mask, smps, acc):
    y1 = y * mask  # A x
    acceleration_rate = acc
    ny = y.shape[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * 0.2 * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * 0.2 * (2 / acceleration_rate)) // 2)

    out_k = grappa(y1.numpy(), y1[:, :, ACS_START_INDEX:ACS_END_INDEX].numpy(), coil_axis=0)
    out_k = torch.tensor(out_k)

    out_k = torch.fft.ifftshift(out_k, [-2, -1])
    x = torch.fft.ifft2(out_k, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    if type(smps) != torch.Tensor:
        smps = torch.tensor(smps)
    x = x * torch.conj(smps)
    x_hat1 = x.sum(0)

    return x_hat1

def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    x = x.astype(np.float32)

    tiff.imwrite(path, x, imagej=True)

def INDEX2_helper(idx, key_):
    file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]

    assert len(file_id_df.index) == 1

    return file_id_df[idx]

INDEX2FILE = lambda idx: INDEX2_helper(idx, 'FILE')

# def INDEX2DROP(idx):
#     ret = INDEX2_helper(idx, 'DROP')
#
#     if ret in ['1', 'true', 'True', 1.0]:
#         return True
#     else:
#         return False

def INDEX2DROP(idx):
    ret = INDEX2_helper(idx, 'DROP')

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True
def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        print(f"data shape: {data.shape}")
        raise ValueError("Invalid shapes.")

    if type(data) != torch.Tensor:
        data = torch.from_numpy(data)

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def zero_pad(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:

    if (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        return data

    if type(data) != torch.Tensor:
        data = torch.from_numpy(data)

    h_pad = max((shape[0] - data.shape[-2]) // 2, 0)
    w_pad = max((shape[1] - data.shape[-1]) // 2, 0)

    data = F.pad(data, (w_pad, w_pad, h_pad, h_pad), mode='constant')

    return data

def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H y

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image, shape: slice, 1, width, height
    """

    # mask^H
    if type(mask) != torch.Tensor:
        mask = torch.tensor(mask)
    if type(y) != torch.Tensor:
        y = torch.tensor(y)

    y = y * mask.unsqueeze(0)

    # F^H (F^(-1))
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    if type(smps) != torch.Tensor:
        smps = torch.tensor(smps)

    x = x * torch.conj(smps)
    x = x.sum(-3)

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    # print(x.shape)
    # print(smps.shape)
    if type(smps) != torch.Tensor:
        smps = torch.from_numpy(smps)
    y = x * smps

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    # mask
    if type(mask) != torch.Tensor:
        mask = torch.tensor(mask)
    
    y = y * mask

    return y

def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1

    if randomly_return:
        mask = mask[np.random.randint(0, acceleration_rate)]
    # else:
    #     mask = mask[0]

    return mask[0], mask[acceleration_rate// 2]


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask
}

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x

def load_real_dataset_handle(
        idx,
        acceleration_rate: int = 1,
        is_return_y_smps_hat: bool = False,
        mask_pattern: str = 'uniformly_cartesian',
        smps_hat_method: str = 'eps',
        cropped = True,
        shape = (320, 320)
):
    # server4 location
    # root_path = os.path.join("/project/cigserver4/export3/a.peng/Dataset/fastmri", 'real')
    # server5 location
    root_path = os.path.join("/project/cigserver5/export1/a.peng/Dataset/fastmri", 'real')
    check_and_mkdir(root_path)

    y_h5 = os.path.join(ROOT_PATH, INDEX2FILE(idx) + '.h5')

    if cropped:
        meas_path = os.path.join(root_path, "acceleration_rate_%d_smps_hat_method_%s_cropped" % (
            acceleration_rate, smps_hat_method))
        check_and_mkdir(meas_path)
        y_cropped_path = os.path.join(meas_path, "y_cropped")
        check_and_mkdir(y_cropped_path)
        y_cropped_h5 = os.path.join(y_cropped_path, INDEX2FILE(idx) + '.h5')

    else:
        meas_path = os.path.join(root_path, "acceleration_rate_%d_smps_hat_method_%s" % (
            acceleration_rate, smps_hat_method))
        check_and_mkdir(meas_path)
        
    gt_smps_h5 = os.path.join(root_path, "acceleration_rate_%d_smps_hat_method_%s_cropped" % (1, smps_hat_method),
                              'smps_hat', INDEX2FILE(idx) + '.h5')

    x_hat_path = os.path.join(meas_path, 'x_hat')
    check_and_mkdir(x_hat_path)
    x_hat_h5 = os.path.join(x_hat_path, INDEX2FILE(idx) + '.h5')

    smps_hat_path = os.path.join(meas_path, 'smps_hat')
    check_and_mkdir(smps_hat_path)
    smps_hat_h5 = os.path.join(smps_hat_path, INDEX2FILE(idx) + '.h5')

    mask_path = os.path.join(meas_path, 'mask')
    check_and_mkdir(mask_path)
    mask_h5 = os.path.join(mask_path, INDEX2FILE(idx) + '.h5')

    grappa_path = os.path.join(meas_path, 'grappa')
    check_and_mkdir(grappa_path)
    grappa_h5 = os.path.join(grappa_path, INDEX2FILE(idx))


    if not os.path.exists(x_hat_h5):

        with h5py.File(y_h5, 'r') as f:
            y = f['kspace'][:]

            # Normalize the kspace to 0-1 region
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))

            if cropped:

                y = torch.tensor(y)

                y = torch.fft.ifftshift(y, [-2, -1])
                x = torch.fft.ifft2(y, norm='ortho')
                x = torch.fft.fftshift(x, [-2, -1])

                x_hat_cropped = center_crop(zero_pad(x, shape), shape)

                y = torch.fft.ifftshift(x_hat_cropped, [-2, -1])
                y = torch.fft.fft2(y, norm='ortho')
                y = torch.fft.fftshift(y, [-2, -1])

                y = y.numpy()

                for i in range(y.shape[0]):
                    y[i] /= np.amax(np.abs(y[i]))
            

        if not os.path.exists(mask_h5):

            _, _, n_x, n_y = y.shape
            if acceleration_rate > 1:
                mask = _mask_fn[mask_pattern]((n_x, n_y), acceleration_rate)[1]

            else:
                mask = np.ones(shape=(n_x, n_y), dtype=np.float32)
            # mask = mask[1]
            mask = np.expand_dims(mask, 0)
            mask = torch.from_numpy(mask)

            with h5py.File(mask_h5, 'w') as f:
                f.create_dataset(name='mask', data=mask)

        else:

            with h5py.File(mask_h5, 'r') as f:
                mask = f['mask'][:]

        if not os.path.exists(smps_hat_h5):

            os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy2'
            os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba2'
            from sigpy.mri.app import EspiritCalib
            from sigpy import Device
            import cupy

            num_slice = y.shape[0]
            iter_ = tqdm.tqdm(range(num_slice), desc='[%d, %s] Generating coil sensitivity map (smps_hat)' % (
                idx, INDEX2FILE(idx)))

            smps_hat = np.zeros_like(y)

            for i in iter_:
                tmp = EspiritCalib(y[i] * np.array(mask), device=Device(2), show_pbar=False).run()
                tmp = cupy.asnumpy(tmp)
                smps_hat[i] = tmp

            with h5py.File(smps_hat_h5, 'w') as f:
                f.create_dataset(name='smps_hat', data=smps_hat)

            tmp = np.ones(shape=smps_hat.shape, dtype=np.uint8)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i, j] = np_normalize_to_uint8(abs(smps_hat[i, j]))
            tiff.imwrite(smps_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        else:
            with h5py.File(smps_hat_h5, 'r') as f:
                smps_hat = f['smps_hat'][:]

        y = torch.from_numpy(y)
        smps_hat = torch.from_numpy(smps_hat)

        x_hat = ftran(y, smps_hat, mask)

        ##added for grappa per slice:
        num_slice = x_hat.shape[0]

        slice_start = 4
        slice_end = num_slice - 5

        print(f"saving grappa for idx {idx}")

    if cropped and not os.path.exists(y_cropped_h5):
        with h5py.File(y_h5, 'r') as f:
            y = f['kspace'][:]

            # Normalize the kspace to 0-1 region
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))

            if cropped:

                y = torch.tensor(y)

                y = torch.fft.ifftshift(y, [-2, -1])
                x = torch.fft.ifft2(y, norm='ortho')
                x = torch.fft.fftshift(x, [-2, -1])

                x_hat_cropped = center_crop(zero_pad(x, shape), shape)

                y = torch.fft.ifftshift(x_hat_cropped, [-2, -1])
                y = torch.fft.fft2(y, norm='ortho')
                y = torch.fft.fftshift(y, [-2, -1])

                y = y.numpy()

                for i in range(y.shape[0]):
                    y[i] /= np.amax(np.abs(y[i]))

        with h5py.File(y_cropped_h5, 'w') as f:
            f.create_dataset(name='y_cropped', data=y)
        print(f"wrote  {y_cropped_h5}")

    if not os.path.exists(grappa_h5):
        check_and_mkdir(grappa_h5)
        with h5py.File(smps_hat_h5, 'r') as f:
            smps_hat = f['smps_hat'][:]
        with h5py.File(mask_h5, 'r', swmr=True) as f:
            mask = f['mask'][0]
        if cropped:
            with h5py.File(y_cropped_h5, 'r') as f:
                y = f['y_cropped'][:]
        else:
            with h5py.File(y_h5, 'r') as f:
                y = f['kspace'][:]
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))
        num_slice = y.shape[0]
        slice_start = 0
        slice_end = num_slice - 5
        y = torch.tensor(y)
        mask = torch.tensor(mask)
        smps_hat = torch.tensor(smps_hat)
        for s in range(slice_start, slice_end):
            grappa_h5_s = os.path.join(grappa_h5, str(s) + '.h5')

            grappa_im = grappa_image(y[s], mask, smps_hat[s], acceleration_rate)

            with h5py.File(grappa_h5_s, 'w') as f:
                f.create_dataset(name='grappa', data=grappa_im)

            grappa_im = torch.view_as_real(grappa_im)
            grappa_im = (grappa_im ** 2).sum(dim=-1).sqrt()

            to_tiff(grappa_im, grappa_h5_s.replace('.h5', '_qc.tiff'))

    ret = {
        'x_hat': x_hat_h5,
        'smps_hat': smps_hat_h5,
        'y': y_h5,
        'mask': mask_h5,
        'grappa': grappa_h5,
        'smps': gt_smps_h5,
    }
    if cropped:
        ret['y_cropped'] = y_cropped_h5

    return ret

class RealMeasurement(Dataset):
    def __init__(
            self,
            idx_list,
            acceleration_rate,
            is_return_y_smps_hat: bool = False,
            mask_pattern: str = 'uniformly_cartesian',
            smps_hat_method: str = 'eps',
            cropped = True,
            input_type = "grappa",
            condition=True
    ):
        self.cropped = cropped
        self.input_type = input_type
        self.__index_maps = []
        self.condition=condition
        for idx in idx_list:
            # if INDEX2DROP(idx):
            #     # print("Found idx=[%d] annotated as DROP" % idx)
            #     continue
            if "AXT2" not in INDEX2FILE(idx):
                continue

            ret = load_real_dataset_handle(
                idx,
                acceleration_rate,
                is_return_y_smps_hat,
                mask_pattern,
                smps_hat_method,
                cropped,
                shape=(320, 320)
            )
            try:
                with h5py.File(ret['x_hat'], 'r') as f:
                    num_slice = f['x_hat'].shape[0]
                    # print(f"success: {ret['x_hat']}")
            except:
                print(f"error: {ret['x_hat']}")

            # if INDEX2SLICE_START(idx) is not None:
            #     slice_start = INDEX2SLICE_START(idx)
            # else:
            slice_start = 0

            # if INDEX2SLICE_END(idx) is not None:
            #     slice_end = INDEX2SLICE_END(idx)
            # else:
            slice_end = num_slice - 5

            for s in range(slice_start, slice_end):
                self.__index_maps.append([ret, s])

            self.acceleration_rate = acceleration_rate

        self.is_return_y_smps_hat = is_return_y_smps_hat

    def __len__(self):
        return len(self.__index_maps)

    def __getitem__(self, item):

        ret, s = self.__index_maps[item]

        # with h5py.File(ret['x_hat'], 'r', swmr=True) as f:
        #     x_hat = f['x_hat'][s]

        with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
            smps_hat = f['smps_hat'][s]
        with h5py.File(ret['smps'], 'r', swmr=True) as f:
            smps_gt = f['smps_hat'][s]
            
            
        if not self.cropped:
            with h5py.File(ret['y'], 'r', swmr=True) as f:
                y_gt = f['kspace'][s]

                # Normalize the kspace to 0-1 region
                y_gt /= np.amax(np.abs(y_gt))
        else:
            with h5py.File(ret['y_cropped'], 'r', swmr=True) as f:
                # print(ret['y_cropped'])
                y_gt = f['y_cropped'][s]

                # Normalize the kspace to 0-1 region
                y_gt /= np.amax(np.abs(y_gt))

        with h5py.File(ret['mask'], 'r', swmr=True) as f:
            mask = f['mask'][0]
        nx = y_gt.shape[-2]
        ny = y_gt.shape[-1]
        identity, _ = _mask_fn['uniformly_cartesian']((nx, ny), 1)

        x_gt =  ftran(y_gt, smps_gt, identity)
        y_hat = fmult(x_gt, smps_hat, mask)
        x_hat = ftran(y_hat, smps_hat, mask)

        with h5py.File(os.path.join(ret['grappa'], str(s)+'.h5'), 'r', swmr=True) as f:
            x_grappa = f['grappa'][:]

        # y_grappa = fmult(x_grappa, smps_hat, identity)

        x_gt = zero_pad(x_gt, [320, 320])
        x_grappa = zero_pad(x_grappa, [320, 320])
        x_hat = zero_pad(x_hat, [320, 320])

        x_gt = center_crop(x_gt, [320, 320])
        x_grappa = center_crop(x_grappa, [320, 320])
        x_hat = zero_pad(x_hat, [320, 320])

        x_gt = torch.view_as_real(x_gt).permute([2, 0, 1]).numpy()
        x_grappa = torch.view_as_real(x_grappa).permute([2, 0, 1]).numpy()
        x_hat = torch.view_as_real(x_hat).permute([2, 0, 1]).numpy()
        
        x_gt = np.float32(x_gt)
        x_hat = np.float32(x_hat)

        # if self.input_type == "grappa":
        #     return x_gt, x_grappa, y, mask, smps_hat
        # elif self.input_type == "raw":
        #     x1 = torch.view_as_real(ftran(y_gt, smps_hat, mask)).permute([2, 0, 1])
        #     return x_gt, x1, y, mask, smps_hat
        # else:
        #     raise NotImplementedError
        if self.condition:
            out_dict = {"masked_img": x_gt}
        else:
            out_dict = {}
        return x_gt, out_dict


