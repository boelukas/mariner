from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from utils.loggers import setup_logger
import time
import os
import sys
from pathlib import Path


class CLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        os.makedirs(self.subcommand + "_results", exist_ok=True)
        setup_logger(
            Path(self.subcommand + "_results", time.strftime("%Y%m%d_%H%M%S") + ".log")
        )
        if self.subcommand != "fit":
            getattr(self.config, self.subcommand).trainer.logger = []
            self.save_config_callback = None

    def after_fit(self) -> None:
        self.trainer.test(self.model, self.datamodule, ckpt_path="last")

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        is_predict = sys.argv[1] == "predict"
        is_fit = sys.argv[1] == "fit"
        is_test = sys.argv[1] == "test"

        parser.add_argument("--comment", type=str)
        parser.add_argument("--batch_size", type=int, default=9)
        parser.add_argument("--train_data_dir", type=str, required=is_fit)
        parser.add_argument("--test_data_dir", type=str, required=is_fit or is_test)
        parser.add_argument("--data_dir", type=str, required=is_predict)  # for predict

        parser.set_defaults({"trainer.enable_model_summary": False})
        parser.set_defaults({"trainer.log_every_n_steps": 10})
        parser.set_defaults({"trainer.check_val_every_n_epoch": 1})
        parser.set_defaults({"trainer.enable_checkpointing": True})
        parser.set_defaults({"trainer.enable_progress_bar": False})
        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {"save_dir": ".", "name": "fit_results"},
                }
            }
        )
        parser.set_defaults(
            {
                "trainer.callbacks": [
                    {
                        "class_path": "lightning.pytorch.callbacks.ModelSummary",
                        "init_args": {"max_depth": "2"},
                    },
                    {"class_path": "utils.callbacks.OverrideEpochStepCallback"},
                    {
                        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "init_args": {
                            "save_top_k": 1,
                            "monitor": "Eval/PSNR",
                            "mode": "max",
                            "filename": "refrr-{epoch:03d}-best",
                        },
                    },
                    {
                        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "init_args": {
                            "every_n_epochs": 1,
                            "save_top_k": 1,
                            "save_last": True,
                            "filename": "refrr-current-{epoch:03d}",
                        },
                    },
                ]
            }
        )

        parser.link_arguments("batch_size", "data.init_args.batch_size")
        parser.link_arguments("batch_size", "model.init_args.batch_size")
        if is_fit:
            parser.link_arguments(
                "train_data_dir", "data.init_args.data_train.init_args.data_root"
            )
        if is_test or is_fit:
            parser.link_arguments(
                "test_data_dir", "data.init_args.data_val.init_args.data_root"
            )
            parser.link_arguments(
                "test_data_dir", "data.init_args.data_test.init_args.data_root"
            )
        if is_predict:
            parser.link_arguments(
                "data_dir", "data.init_args.data_pred.init_args.data_root"
            )
        parser.link_arguments("comment", "trainer.logger.init_args.version")


def cli_main() -> None:
    CLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
