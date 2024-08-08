"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger, util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.pmri_fastmri_brain import RealMeasurement
from torch.utils.data import DataLoader, Dataset


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    dataset = RealMeasurement(idx_list=range(1301, 1377), acceleration_rate=8, is_return_y_smps_hat = True, cropped = True, input_type="grappa", condition=False)
    loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    logger.log("creating samples...")
    all_images = []
    psnrs = []
    for loader_itr, (truth, model_kwargs) in enumerate(loader):

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 2, 320, 320),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        truth = truth.to(dist_util.dev())
        sampled_psnr = util.compare_psnr(util.complex_to_im(sample), util.complex_to_im(truth))
        logger.log(f"sampled_psnr: {sampled_psnr:.4f}")
        save_path=os.path.join(logger.get_dir(), "sample.tiff")
        util.batch_save_as_tiff(util.complex_to_im(sample), save_path, has_batch=True)
        
        
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        truth = truth.cpu()
        
    
        psnrs.append(sampled_psnr.cpu())

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log(f"average psnr: {np.array(psnrs).mean()}")
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="/project/cigserver5/export1/a.peng/guided-diffusion-mri/logs/openai-2024-07-17-18-05-08-063842/model340000.pt",
        beta_scale=0.15
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
