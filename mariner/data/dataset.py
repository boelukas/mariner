import os
from PIL import Image
import numpy as np
import glob
import random
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from data.padding import pad_to_multiple_of


class REFRR(Dataset):
    def __init__(
        self,
        data_root: str = "",
        stage: str = "train",
        data_augmentation: bool = True,
        random_ref_prob: float = 0.0,
        black_ref_image_prob: float = 0.0,
        use_gt_as_ref: bool = False,
        ref_level: int = 1,
        scale: int = 4,
    ) -> None:
        """Creates a REFRR dataset.
        :param data_root: Directory where the dataset is downloaded.
        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. The stage this dataset is for.
        :param data_augmentation: If data augmentation should be used.
        :param use_gt_as_ref: If the ground truth should be used as reference.
        :param scale: Scale used to downsample the input and ref images. (Needed for compatibility with MASA)
        """
        self.train_in_list = None
        self.train_gt_list = None
        self.train_ref_lists = None
        self.test_in_list = None
        self.test_gt_list = None
        self.test_ref_lists = None
        self.pred_in_list = None
        self.pred_ref_list = None
        self.data_root = data_root
        self.stage = stage
        self.data_augmentation = data_augmentation
        self.use_gt_as_ref = use_gt_as_ref
        self.ref_level = ref_level
        self.scale = scale
        self.random_ref_prob = random_ref_prob
        self.black_ref_image_prob = black_ref_image_prob

    def load_lists(self) -> None:
        if self.stage == "fit" and not self.train_gt_list:
            self.train_in_list = sorted(
                glob.glob(str(Path(self.data_root, "input", "*.png")))
            )
            self.train_gt_list = sorted(
                glob.glob(str(Path(self.data_root, "gt", "*.png")))
            )
            self.train_ref_lists = []
            if len(self.train_in_list) > 0:
                first_im_name = Path(self.train_in_list[0]).stem
                num_references = len(
                    glob.glob(
                        str(
                            Path(
                                self.data_root,
                                "ref",
                                first_im_name + "*",
                            )
                        )
                    )
                )
                for i in range(num_references):
                    self.train_ref_lists.append(
                        sorted(
                            glob.glob(
                                str(
                                    Path(
                                        self.data_root,
                                        "ref",
                                        "*_" + f"{i:02d}" + ".png",
                                    )
                                )
                            )
                        )
                    )

        if self.stage in ["test", "validation"] and not self.test_gt_list:
            self.test_in_list = sorted(
                glob.glob(str(Path(self.data_root, "input", "*.png")))
            )
            self.test_gt_list = sorted(
                glob.glob(str(Path(self.data_root, "gt", "*.png")))
            )
            self.test_ref_lists = []
            if len(self.test_in_list) > 0:
                first_im_name = Path(self.test_in_list[0]).stem
                num_references = len(
                    glob.glob(
                        str(
                            Path(
                                self.data_root,
                                "ref",
                                first_im_name + "*",
                            )
                        )
                    )
                )
                for i in range(num_references):
                    self.test_ref_lists.append(
                        sorted(
                            glob.glob(
                                str(
                                    Path(
                                        self.data_root,
                                        "ref",
                                        "*_" + f"{i:02d}" + ".png",
                                    )
                                )
                            )
                        )
                    )
        if self.stage == "predict" and not self.pred_in_list:
            self.pred_in_list = sorted(
                glob.glob(str(Path(self.data_root, "input", "*.png")))
            )
            self.pred_ref_list = sorted(
                glob.glob(str(Path(self.data_root, "ref", "*.png")))
            )

    def gt_images(self) -> list[str]:
        if self.stage == "fit":
            return self.train_gt_list
        else:
            return self.test_gt_list

    def ref_images(self) -> list[str]:
        if self.stage == "fit":
            return self.train_ref_lists[self.ref_level]
        elif self.stage in ["test", "validation"]:
            return self.test_ref_lists[self.ref_level]
        else:
            return self.pred_ref_list

    def __len__(self) -> int:
        self.load_lists()
        if self.stage == "fit":
            return len(self.train_in_list)
        elif self.stage in ["test", "validation"]:
            return len(self.test_in_list)
        else:
            return len(self.pred_in_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.load_lists()

        if self.stage == "fit":
            input = cv2.imread(self.train_in_list[idx])
            gt = cv2.imread(self.train_gt_list[idx])
            if self.use_gt_as_ref:
                ref = cv2.imread(self.train_gt_list[idx])
            else:
                if random.random() < self.random_ref_prob:
                    rand_ref_level = random.randint(0, len(self.train_ref_lists) - 1)
                    ref = cv2.imread(self.train_ref_lists[rand_ref_level][idx])
                else:
                    ref = cv2.imread(self.train_ref_lists[self.ref_level][idx])
            if random.random() < self.black_ref_image_prob:
                ref = np.zeros_like(ref)
        elif self.stage in ["test", "validation"]:
            input = cv2.imread(self.test_in_list[idx])
            gt = cv2.imread(self.test_gt_list[idx])
            if self.use_gt_as_ref:
                ref = cv2.imread(self.test_gt_list[idx])
            else:
                ref = cv2.imread(self.test_ref_lists[self.ref_level][idx])
        elif self.stage == "predict":
            input = cv2.imread(self.pred_in_list[idx])
            ref = cv2.imread(self.pred_ref_list[idx])

        # pad ref to be multiple of 16
        ref_pad, ref_padnums = pad_to_multiple_of(ref, 16, mode="bottom-right")

        # pad in and gt to be mutiple of 16 or 32 for test, validation and prediction
        if self.stage == "fit":
            input_pad, input_padums = pad_to_multiple_of(input, 16, mode="bottom-right")
            gt_pad, gt_padums = pad_to_multiple_of(gt, 16, mode="bottom-right")
        elif self.stage in ["test", "validation"]:
            input_pad, input_padums = pad_to_multiple_of(input, 32, mode="even")
            gt_pad, gt_padums = pad_to_multiple_of(gt, 32, mode="even")
        elif self.stage in "predict":
            input_pad, input_padums = pad_to_multiple_of(input, 32, mode="even")

        if self.stage == "fit" and self.data_augmentation:
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-20, 20)
            if random.random() < 0.5:
                ref_pad = cv2.convertScaleAbs(ref_pad, alpha=alpha, beta=beta)
                ref_pad = np.clip(ref_pad, 0, 255)
            else:
                input_pad = cv2.convertScaleAbs(input_pad, alpha=alpha, beta=beta)
                input_pad = np.clip(input_pad, 0, 255)
                gt_pad = cv2.convertScaleAbs(gt_pad, alpha=alpha, beta=beta)
                gt_pad = np.clip(gt_pad, 0, 255)

        h, w, _ = input_pad.shape
        input_lr_pad = np.array(
            Image.fromarray(input_pad).resize(
                (w // self.scale, h // self.scale), Image.BICUBIC
            )
        )

        h, w, _ = ref_pad.shape
        ref_lr_pad = np.array(
            Image.fromarray(ref_pad).resize(
                (w // self.scale, h // self.scale), Image.BICUBIC
            )
        )
        sample = {
            "IN": input_pad,
            "REF": ref_pad,
            "IN_LR": input_lr_pad,
            "REF_LR": ref_lr_pad,
        }

        if self.stage != "predict":
            sample["GT"] = gt_pad

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.0
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        if self.stage != "predict":
            sample["GT_pad_nums"] = torch.tensor(gt_padums)
        sample["IN_pad_nums"] = torch.tensor(input_padums)
        sample["REF_pad_nums"] = torch.tensor(ref_padnums)

        if self.stage in ["test", "validation"]:
            sample["name"] = os.path.basename(self.test_gt_list[idx])
        if self.stage == "predict":
            sample["name"] = os.path.basename(self.pred_in_list[idx])
        if self.stage == "fit":
            sample["name"] = os.path.basename(self.train_in_list[idx])

        return sample
