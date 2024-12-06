import numpy as np
import torch

"""
This file is collecting of augmentation functions used in the time series analyzing paper.
"""


def DataTransform(sample, config, type="strong"):
    assert type in ["weak", "strong"], f"{type} is not supported."
    if type == "weak":
        weak_aug = scaling(sample, sigma=config["weak_augmentation"]["jitter_scale_ratio"])
        return weak_aug
    elif type == "strong":
        strong_aug = jitter(
            permutation(
                sample,
                max_segments=config["strong_augmentation"]["max_seg"],
                seg_mode=config["strong_augmentation"]["seg_mode"],
            ),
            config["strong_augmentation"]["jitter_ratio"],
        )
        return strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    x = x.cpu().numpy()
    factor = np.random.normal(loc=2.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # import random

            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()

            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
