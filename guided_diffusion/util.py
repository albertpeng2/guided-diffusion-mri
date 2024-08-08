import numpy as np
import os
import tifffile as tiff
import torch

def compare_mse(img_test, img_true, size_average=True):
    img_test = (img_test - img_test.min())/(img_test.max() - img_test.min())
    img_true = (img_true - img_true.min())/(img_true.max() - img_true.min())

    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff

def batch_save_as_tiff(images, path, has_batch = True):
    # images: has to be [batch, height, width] or [batch, channel(2), height, width]
    # path: path to save
    #output: images saved with horizontal stack
    if (len(images.shape) == 4 and has_batch) or (not has_batch and len(images.shape) == 3):
        images = complex_to_im(images)
    if len(images.shape) == 2 and not has_batch:
        images = images.unsqueeze(0)
    if len(images.shape) != 3:
        print(f"wrong image shape: {images.shape}")
    #normalize each image
    for i in range(images.shape[0]):
        images[i, :] -= torch.amin(images[i, :])
        images[i, :] /= torch.amax(images[i, :])

    images = images.mul(255).add_(0.5).clamp_(0, 255)

    im_list = [images[i] for i in range(images.shape[0])]
    recon_img = torch.hstack(im_list).to("cpu", torch.uint8).numpy()
    tiff.imwrite(os.path.join(path), recon_img, imagej=True)
    
def compare_psnr_im(img_test, img_true):
    img_test = (img_test - img_test.min())/(img_test.max() - img_test.min())
    img_true = (img_true - img_true.min())/(img_true.max() - img_true.min())

    return 10 * torch.log10((1 ** 2) / compare_mse(img_test, img_true))

def compare_mse(img_test, img_true, size_average=True):
    img_test = (img_test - img_test.min())/(img_test.max() - img_test.min())
    img_true = (img_true - img_true.min())/(img_true.max() - img_true.min())

    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

def complex_to_im(img: torch.Tensor) -> torch.Tensor:
    # input: [:, Channel=2, H, W] Tensor, where the two channels are real and complex values
    # output: [:, H, W] Tensor
    assert img.shape[-3] == 2   # is complex
    img = torch.sum((img ** 2), dim=-3).sqrt()
    return img