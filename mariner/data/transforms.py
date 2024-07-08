import numpy as np
import torch


class BatchTransform:
    """Transforms a batch. Used for data augmentation"""

    def __init__(self, start_epoch: int = 0, end_epoch: int = -1) -> None:
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def __call__(
        self, batch: dict[str, torch.Tensor], epoch: int
    ) -> dict[str, torch.Tensor]:
        """Takes a batch, applies a transform, returns the transformed batch."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def is_active(self, epoch: int) -> bool:
        if self.start_epoch <= epoch:
            if epoch <= self.end_epoch or self.end_epoch == -1:
                return True
        return False


class BatchTransformMASA(BatchTransform):
    def __init__(self, start_epoch: int = 0, end_epoch: int = -1) -> None:
        super(BatchTransformMASA, self).__init__(start_epoch, end_epoch)

    def __call__(
        self, batch: dict[str, torch.Tensor], epoch: int
    ) -> dict[str, torch.Tensor]:
        batch_samples = batch
        if self.is_active(epoch):
            # flip
            if np.random.randint(0, 2) == 1:  # Horizontal flip
                batch_samples["GT"] = torch.flip(
                    batch_samples["GT"], dims=[2]
                )  # [N, C, H, W]
                batch_samples["IN"] = torch.flip(batch_samples["IN"], dims=[2])
                batch_samples["IN_LR"] = torch.flip(batch_samples["IN_LR"], dims=[2])
            if np.random.randint(0, 2) == 1:
                batch_samples["REF"] = torch.flip(batch_samples["REF"], dims=[2])
                batch_samples["REF_LR"] = torch.flip(batch_samples["REF_LR"], dims=[2])
            if np.random.randint(0, 2) == 1:  # Vertical flip
                batch_samples["GT"] = torch.flip(batch_samples["GT"], dims=[3])
                batch_samples["IN"] = torch.flip(batch_samples["IN"], dims=[3])
                batch_samples["IN_LR"] = torch.flip(batch_samples["IN_LR"], dims=[3])
            if np.random.randint(0, 2) == 1:
                batch_samples["REF"] = torch.flip(batch_samples["REF"], dims=[3])
                batch_samples["REF_LR"] = torch.flip(batch_samples["REF_LR"], dims=[3])
            # rotate
            if np.random.randint(0, 2) == 1:
                k = np.random.randint(0, 4)
                batch_samples["GT"] = torch.rot90(batch_samples["GT"], k, dims=[2, 3])
                batch_samples["IN"] = torch.rot90(batch_samples["IN"], k, dims=[2, 3])
                batch_samples["IN_LR"] = torch.rot90(
                    batch_samples["IN_LR"], k, dims=[2, 3]
                )
            if np.random.randint(0, 2) == 1:
                k = np.random.randint(0, 4)
                batch_samples["REF"] = torch.rot90(batch_samples["REF"], k, dims=[2, 3])
                batch_samples["REF_LR"] = torch.rot90(
                    batch_samples["REF_LR"], k, dims=[2, 3]
                )
        return batch_samples


class BatchTransformAllOrNothing(BatchTransform):
    def __init__(self, start_epoch: int = 0, end_epoch: int = -1) -> None:
        super(BatchTransformAllOrNothing, self).__init__(start_epoch, end_epoch)

    def __call__(
        self, batch: dict[str, torch.Tensor], epoch: int
    ) -> dict[str, torch.Tensor]:
        batch_samples = batch
        if self.is_active(epoch):
            # Vertical flip
            if np.random.randint(0, 2) == 1:
                batch_samples["GT"] = torch.flip(batch_samples["GT"], dims=[3])
                batch_samples["IN"] = torch.flip(batch_samples["IN"], dims=[3])
                batch_samples["IN_LR"] = torch.flip(batch_samples["IN_LR"], dims=[3])
                batch_samples["REF"] = torch.flip(batch_samples["REF"], dims=[3])
                batch_samples["REF_LR"] = torch.flip(batch_samples["REF_LR"], dims=[3])
            # rotate
            if np.random.randint(0, 2) == 1:
                k = np.random.randint(0, 4)
                batch_samples["GT"] = torch.rot90(batch_samples["GT"], k, dims=[2, 3])
                batch_samples["IN"] = torch.rot90(batch_samples["IN"], k, dims=[2, 3])
                batch_samples["IN_LR"] = torch.rot90(
                    batch_samples["IN_LR"], k, dims=[2, 3]
                )
                batch_samples["REF"] = torch.rot90(batch_samples["REF"], k, dims=[2, 3])
                batch_samples["REF_LR"] = torch.rot90(
                    batch_samples["REF_LR"], k, dims=[2, 3]
                )
        return batch_samples
