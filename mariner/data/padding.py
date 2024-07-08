import torch
import math
import numpy as np
import cv2

from utils.visualizations import Visualization


def pad_to_multiple_of(
    x: np.ndarray, multiple: int, mode: str = "even"
) -> tuple[np.ndarray, list[int]]:
    """Pads `"x"` to be a multiple of `"multiple"`.
    :param x: [H, W, C] The input array to be padded.
    :param multiple: The factor, which the new size needs to be a multiple of.
    :param mode: Either `"even"`, `"bottom-right"`. Whether to apply even padding evenly around the array or only at the bottom right corner.
    """
    h, w, c = x.shape
    if h % multiple != 0 or w % multiple != 0:
        h_new = math.ceil(h / multiple) * multiple
        w_new = math.ceil(w / multiple) * multiple
        if mode == "even":
            pad_HR_t = (h_new - h) // 2
            pad_HR_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_HR_l = (w_new - w) // 2
            pad_HR_r = (w_new - w) // 2 + (w_new - w) % 2
        elif mode == "bottom-right":
            pad_HR_t, pad_HR_d, pad_HR_l, pad_HR_r = 0, h_new - h, 0, w_new - w
    else:
        return x, [0, 0, 0, 0]

    x_pad = cv2.copyMakeBorder(
        x, pad_HR_t, pad_HR_d, pad_HR_l, pad_HR_r, cv2.BORDER_REPLICATE
    )

    return x_pad, [pad_HR_t, pad_HR_d, pad_HR_l, pad_HR_r]


def remove_padding(
    batch: dict[str, torch.Tensor],
    net_outs=list[torch.Tensor],
    visualization: Visualization = None,
) -> tuple[dict[str, torch.Tensor], list[torch.tensor]]:
    """Removes the padding from batch['GT', 'IN', 'REF'], all outputs and adjusts the visualization.
    :param batch: The batch, as returned by the dataset. Each batch['GT', 'IN', 'REF'] needs to have a <key>_pad_nums element with the numbers that the tensor was padded with.
    :param net_outs: The ouputs of the network.
    :param visualization: optional, a visualization where the src and dst offset has to be adjusted when padding is removed.
    :return: The unpadded batch and outputs
    """
    batch_unpad = {}
    for key in ["GT", "IN", "REF"]:
        # Predictions have no GT
        if key not in batch:
            continue
        # pad_t, pad_d, pad_l, pad_r
        padnums = batch[key + "_pad_nums"][0]
        _, _, H, W = batch[key].shape
        batch_unpad[key] = batch[key][
            :, :, padnums[0] : H - padnums[1], padnums[2] : W - padnums[3]
        ]
    outputs_unpad = []
    for output in net_outs:
        padnums = batch["IN_pad_nums"][0]
        _, _, H, W = output.shape
        output_unpad = output[
            :, :, padnums[0] : H - padnums[1], padnums[2] : W - padnums[3]
        ]
        outputs_unpad.append(output_unpad)

    if visualization:
        src_padnums = batch["IN_pad_nums"][0]
        visualization.src_offset[0] -= src_padnums[0]  # top
        visualization.src_offset[1] -= src_padnums[2]  # left

        dst_padnums = batch["REF_pad_nums"][0]
        visualization.dst_offset[0] -= dst_padnums[0]  # top
        visualization.dst_offset[1] -= dst_padnums[2]  # left

    return batch_unpad, outputs_unpad
