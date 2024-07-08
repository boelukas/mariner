import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import logging
from pathlib import Path
import torchvision
import os
import time
from collections import Counter
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torchmetrics import MaxMetric, MeanMetric
from models.components.losses import Loss
from utils.scores import calculate_masa_psnr_ssim
from utils.visualizations import batch_to_images
from data.padding import remove_padding


class MARINER(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: list[Loss],
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        tb_log_images: list[str],
        compile: bool = False,
        log_all_n_steps: int = 10,
        batch_size: int = 9,
        ckpt_epochs: list[int] = [],
        iterations: int = 1,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.compile = compile
        self.tb_log_images = tb_log_images
        self.vis_images = None
        self.log_all_n_steps = log_all_n_steps
        self.batch_size = batch_size
        self.ckpt_epochs = ckpt_epochs
        self.iterations = iterations

        self.loss = torch.nn.ModuleList(loss)
        self.test_l1_criterion = nn.L1Loss()

        self.train_loss = MeanMetric()

        self.val_psnr = MeanMetric()
        self.test_psnr = MeanMetric()

        self.train_ssim = MeanMetric()
        self.val_ssim = MeanMetric()
        self.test_ssim = MeanMetric()
        self.test_l1 = MeanMetric()

        self.val_psnr_best = MaxMetric()

        # Uncomment for plotting the computational graph
        # self.example_input_array = (torch.Tensor(9, 3, 160, 160), torch.Tensor(9, 3, 160, 160), torch.Tensor(9, 3, 40, 40))

    def forward(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        inputs = [batch["IN"]]
        outputs = []
        for i in range(self.iterations):
            batch["IN"] = inputs[i]
            out = self.net(batch)
            inputs.append(out.clamp(min=0, max=1))
            outputs.append(out)
        batch["IN"] = inputs[0]
        return outputs

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_psnr.reset()
        logging.info(
            "%d training samples" % (len(self.trainer.train_dataloader.dataset))
        )
        logging.info("training started")

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step. In all other steps, the padding is removed before calculating scores or doing visualizations.
        Not in training, because in CUFED some references are 159x160 and therefore not the same shape as the gt. With REFRR the reference has always the same shape as the gt.
        The adversarial loss requires gt and out to be 160 x 160.
        """
        outs = self.forward(batch)

        losses = [
            self.calculate_loss(out, batch["GT"], "Train", iter)
            for iter, out in enumerate(outs)
        ]

        # Adds the matching dict elements together in the list of dicts
        loss = Counter()
        for iter_loss in losses:
            loss.update(iter_loss)

        self.train_loss(loss["total"])

        if batch_idx % self.log_all_n_steps == 0:
            log_info = "epoch:%03d step:%04d  " % (
                self.trainer.current_epoch,
                batch_idx,
            )
            for name, val in loss.items():
                log_info += "%s:%.06f " % (name, val.item())

            for tr in self.trainer.datamodule.transforms:
                if tr.is_active(self.trainer.current_epoch):
                    log_info += "Aug: %s " % (tr)
            logging.info(log_info)

        if batch_idx == 0:
            images = batch_to_images(
                batch,
                outs,
                ["IN", "REF", "GT", "OUT"],
                self.net.attention.visualization,
            )
            save_names = [
                Path(
                    self.trainer.log_dir,
                    "vis",
                    "vis_%d_%d.png" % (self.trainer.current_epoch, j),
                )
                for j in range(batch["IN"].shape[0])
            ]
            os.makedirs(Path(self.trainer.log_dir, "vis"), exist_ok=True)
            for image, name in zip(images, save_names):
                image.save(name)
        return loss["total"]

    def on_train_epoch_start(self) -> None:
        logging.info("current_lr: %f" % (self.optimizers().param_groups[0]["lr"]))

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.trainer.current_epoch in self.ckpt_epochs:
            self.trainer.save_checkpoint(
                Path(
                    self.trainer.log_dir,
                    "checkpoints",
                    "refrr-epoch=%03d.ckpt" % self.trainer.current_epoch,
                )
            )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input, ground truth and reference tensor
        :param batch_idx: The index of the current batch.
        """
        outs = self.forward(batch)

        batch_unpad, outs_unpad = remove_padding(batch, outs)
        gt = batch_unpad["GT"]
        out = outs_unpad[-1]

        psnr_masa, ssim_masa = calculate_masa_psnr_ssim(out, gt)
        self.val_psnr(psnr_masa)
        self.val_ssim(ssim_masa)

        self.log(
            "Eval/PSNR",
            self.val_psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Eval/SSIM",
            self.val_ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def on_validation_start(self) -> None:
        logging.info("start validation")
        logging.info("%d val samples" % (len(self.trainer.val_dataloaders.dataset)))
        if not self.vis_images:
            self.vis_images = self.find_vis_images(
                self.tb_log_images, self.trainer.val_dataloaders.dataset
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        self.val_psnr_best(psnr)
        logging.info("psnr:%.06f   ssim:%.06f " % (psnr, ssim))
        if psnr == self.val_psnr_best.compute():
            logging.info("best_psnr:%.06f " % (psnr))
        if self.vis_images:
            vis_outs = self.forward(self.vis_images)
            batch_unpad, outs_unpad = remove_padding(
                self.vis_images, vis_outs, self.net.attention.visualization
            )
            images = batch_to_images(
                batch_unpad,
                outs_unpad,
                ["IN", "REF", "OUT"],
                self.net.attention.visualization,
            )
            for image, name in zip(images, self.vis_images["name"]):
                self.logger.experiment.add_image(
                    "VIS/" + name, pil_to_tensor(image), self.trainer.current_epoch
                )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict) containing the input, ground truth and reference tensor
        :param batch_idx: The index of the current batch.
        """
        outs = self.forward(batch)
        image_name = batch["name"][0]
        batch_unpad, outs_unpad = remove_padding(
            batch, outs, self.net.attention.visualization
        )

        out = outs_unpad[-1]
        gt = batch_unpad["GT"]

        self.test_l1(self.test_l1_criterion(out, gt))
        psnr_masa, ssim_masa = calculate_masa_psnr_ssim(out, gt)
        self.test_psnr(psnr_masa)
        self.test_ssim(ssim_masa)
        self.log(
            "Test/PSNR",
            self.test_psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Test/SSIM",
            self.test_ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Test/L1",
            self.test_l1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        save_path_all = Path(self.trainer.default_root_dir, "test_results", "out_all")
        save_path_only_tb = Path(
            self.trainer.default_root_dir, "test_results", "out_only_tb"
        )
        os.makedirs(save_path_all, exist_ok=True)
        os.makedirs(save_path_only_tb, exist_ok=True)

        path_all = Path(save_path_all, image_name)
        path_only_tb = Path(save_path_only_tb, image_name)

        image = batch_to_images(
            batch_unpad, [out], ["IN", "REF", "OUT"], self.net.attention.visualization
        )[0]
        logging.info("saving %d_th image: %s" % (batch_idx, image_name))
        image.save(path_all)
        if image_name in self.tb_log_images:
            logging.info("saving %d_th image: %s" % (batch_idx, image_name))
            image.save(path_only_tb)

    def on_test_start(self) -> None:
        logging.info("start testing")
        self.vis_images = self.find_vis_images(
            self.tb_log_images, self.trainer.test_dataloaders.dataset
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def on_predict_start(self) -> None:
        self.prediction_path = Path(
            self.trainer.predict_dataloaders.dataset.data_root, "out"
        )
        os.makedirs(self.prediction_path, exist_ok=True)
        logging.info("start predicting")
        logging.info("saving predictions to: %s" % self.prediction_path)
        self.predict_start_time = time.time()

    def on_predict_end(self) -> None:
        datase_size = len(self.trainer.predict_dataloaders.dataset)
        end_time = time.time()
        logging.info(
            "Avg prediction time: %f s"
            % ((end_time - self.predict_start_time) / datase_size)
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outs = self.forward(batch)
        batch_unpad, outs_unpad = remove_padding(batch, outs)
        out = outs_unpad[-1]
        image_name = batch["name"][0]
        out_path = Path(self.prediction_path, image_name)
        out_img = out[0].flip(dims=(0,)).clamp(0.0, 1.0)
        torchvision.utils.save_image(out_img, out_path)
        logging.info("saving %d_th image: %s" % (batch_idx, image_name))
        return out

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        logging.info("the init lr: %f" % (optimizer.param_groups[0]["lr"]))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: dict[str, any]) -> None:
        "Objects to include in checkpoint file"
        checkpoint["val_psnr_best"] = self.val_psnr_best

    def on_load_checkpoint(self, checkpoint: dict[str, any]) -> None:
        "Objects to retrieve from checkpoint file"
        self.val_psnr_best(checkpoint["val_psnr_best"].compute())

    def calculate_loss(
        self, out: torch.Tensor, gt: torch.Tensor, stage: str, current_iteration: int
    ) -> dict[str, torch.Tensor]:
        """Checks for every loss if the loss is active in the current epoch and iteration. If yes, computes this loss and multiplies it with the weight.
        Adds all active losses together. For the adv loss, only the generator loss is added but both losses are logged to TB.
        :param out: [N, C, H, W] Network output
        :param gt: [N, C, H, W] Ground truth
        :param stage: Current stage. E.g Train. Is prepended to the TB logs.
        :param current_iteration: The current iteration for losses that are only active in certain iterations.
        :return: Dict with key loss_name and value loss_value. Additionally the total key with the weighted loss sum is present.
        """
        l = []
        loss_dict = {}
        for lo in self.loss:
            if lo.active(self.current_epoch, current_iteration):
                loss_names = lo.tb_name()
                if "adv_loss" in loss_names:
                    g_loss, d_loss = lo(out, gt)
                    if stage == "Train":
                        self.log(
                            stage + "/" + loss_names[0],
                            g_loss * lo.weight,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            batch_size=self.batch_size,
                        )
                        self.log(
                            stage + "/" + loss_names[1],
                            d_loss * lo.weight,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            batch_size=self.batch_size,
                        )
                    l.append(g_loss * lo.weight)
                    loss_dict[loss_names[0]] = g_loss * lo.weight

                else:
                    w_loss_val = lo(out, gt) * lo.weight
                    if stage == "Train":
                        self.log(
                            stage + "/" + loss_names[0],
                            w_loss_val,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            batch_size=self.batch_size,
                        )
                    l.append(w_loss_val)
                    loss_dict[loss_names[0]] = w_loss_val

        total_loss = (torch.stack(l, dim=0)).sum()
        loss_dict["total"] = total_loss
        return loss_dict

    def find_vis_images(
        self, tb_log_images: list[str], dataset: Dataset
    ) -> dict[str, torch.Tensor]:
        if len(tb_log_images) == 0:
            return {}
        im_idxs = [
            i
            for i, x in enumerate(dataset.gt_images())
            if x.endswith(tuple(tb_log_images))
        ]
        if len(im_idxs) != len(tb_log_images):
            logging.warn(f"Some tb_log_images were not found.")
            return {}

        samples = [dataset[idx] for idx in im_idxs]
        vis_images = {}
        for key in samples[0].keys():
            if torch.is_tensor(samples[0][key]):
                vis_images[key] = torch.stack([s[key] for s in samples]).to(self.device)
            else:
                vis_images[key] = [s[key] for s in samples]
        return vis_images
