"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger, script_util, util
from guided_diffusion.pmri_fastmri_brain import RealMeasurement
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.utils.data import DataLoader, Dataset

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    dataset = RealMeasurement(idx_list=range(1301, 1377), acceleration_rate=8, is_return_y_smps_hat = True, cropped = True, input_type="grappa")
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
            (args.batch_size, 2, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        truth = truth.to(dist_util.dev())
        sampled_psnr = util.compare_psnr(util.complex_to_im(sample), util.complex_to_im(truth))
        grappa_psnr = util.compare_psnr(util.complex_to_im(model_kwargs["masked_img"]), util.complex_to_im(truth))
        
        sampled_psnr = util.compare_psnr(util.complex_to_im(sample), util.complex_to_im(truth))
        logger.log(f"sampled_psnr: {sampled_psnr:.4f}")
        save_path=os.path.join(logger.get_dir(), f"sample.tiff_{loader_itr}")
        util.batch_save_as_tiff(util.complex_to_im(sample), save_path, has_batch=True)
        
        logger.log(f"grappa img: {grappa_psnr:.4f}            sampled_psnr: {sampled_psnr:.4f}")
        
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


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []
                
def load_superres_data(batch_size, acceleration=8, cropped=True, input_type="raw"):
    dataset = RealMeasurement(idx_list=range(1301, 1377), acceleration_rate=acceleration, is_return_y_smps_hat = True, cropped = cropped, input_type=input_type, condition=True)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    while True:
        yield from loader


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        base_samples="",
        large_size=320,
        progress=True,
        model_path="/project/cigserver5/export1/a.peng/guided-diffusion-mri/logs/openai-2024-07-23-12-31-58-349689/model1200000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
