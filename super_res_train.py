"""
Train a super-resolution model.
"""

import argparse
import sys, os
import numpy as np
sys.path.insert(0, "../")

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
# from guided_diffusion.image_datasets import load_data
# from guided_diffusion.mri_dataset import load_data
from guided_diffusion.pmri_fastmri_brain import RealMeasurement
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    add_dict_to_argparser,
)
from torch.utils.data import DataLoader, Dataset
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    logger.configure(args=args)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(
        batch_size = args.batch_size,
        small_size = (128, 128),
        class_cond=False,
        acceleration=args.acceleration,
        cropped=args.cropped,
        input_type=args.input_type
    )


    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

# original
# def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
#     data = load_data(
#         data_dir=data_dir,
#         batch_size=batch_size,
#         image_size=large_size,
#         class_cond=class_cond,
#     )
#     for large_batch, model_kwargs in data:
#         model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
#         yield large_batch, model_kwargs
#

def load_superres_data(batch_size, small_size, class_cond=False, acceleration=8, cropped=True, input_type="raw"):
    dataset = RealMeasurement(idx_list=range(1, 1300), acceleration_rate=acceleration, is_return_y_smps_hat = True, cropped = cropped, input_type=input_type)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    # for large_batch, model_kwargs in loader:
    #     # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
    #     yield large_batch, model_kwargs
    while True:
        yield from loader


def create_argparser():
    defaults = dict(
        acceleration=8,
        cropped=True,
        input_type="raw",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        wandb_api_key="fe47e48c41d8c2349446e130f510040c843ae8de",
        wandb_user="albertpeng2",
        name="ddpm",
        beta_scale=0.1
    )
    
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
