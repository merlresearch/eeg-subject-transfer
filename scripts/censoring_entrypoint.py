# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import sys
from pathlib import Path
from pprint import pformat

from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.censored_models import CensoredModel
from src.data import THUDataModule
from src.utils import PROJECT_PATH, add_start_stop_flags, get_git_hash


@logger.catch(onerror=lambda _: sys.exit(1))
def main(args):
    logger.remove()
    logger.add(sys.stdout)
    save_dir = PROJECT_PATH / "results" / args.results_dir
    save_dir.mkdir(exist_ok=True, parents=True)
    seed_everything(args.seed, workers=True)
    # NOTE - we provide all args to all constructors. Names that appear in init args will be captured
    datamodule = THUDataModule(**vars(args))
    datamodule.setup()
    model = CensoredModel(
        num_classes=datamodule.num_classes,
        input_shape=datamodule.input_shape,
        class_weights=datamodule.class_weights,
        n_train_subj=datamodule.n_train_subj,
        **vars(args),
    )
    tb_logger = TensorBoardLogger(
        # Files go to save_dir/name/version/sub_dir
        save_dir=save_dir,
        name=args.name,
        # version=None -> uses version_0, version_1, etc; we need version="" to omit
        # sub_dir=None will omit this part by default
        version="",
        flush_secs=10,
        log_graph=True,
        default_hp_metric=False,
    )
    csv_logger = CSVLogger(
        # Files go to save_dir/name/version
        save_dir=save_dir,
        name=tb_logger.name,
        version="",
    )

    callbacks = []
    filename_template = "epoch={epoch}-step={step}-val_bal_acc={val/bal_acc:.3f}"
    if args.use_val_set:
        best_ckpt = ModelCheckpoint(
            save_top_k=1,
            monitor="val/bal_acc",
            mode="max",
            filename="best__" + filename_template,
            auto_insert_metric_name=False,
        )
        callbacks.append(best_ckpt)
    last_ckpt = ModelCheckpoint(
        filename="last__" + filename_template,
        auto_insert_metric_name=False,
    )
    callbacks.append(last_ckpt)
    # Create trainer
    trainer = Trainer.from_argparse_args(
        args,
        num_sanity_val_steps=0,
        inference_mode=True,  # use torch.inference_mode instead of torch.no_grad
        deterministic=True,
        logger=[tb_logger, csv_logger],
        # NOTE - order of ckpt callbacks is important. when trainer has multiple ckpt callbacks,
        # and we use `.text(ckpt_path="best")`, the first ckpt callback will be used
        callbacks=callbacks,
    )
    if not args.fast_dev_run:
        logger.add(Path(trainer.logger.log_dir) / "log.txt")

    logger.info(f"Git hash: {args.git_hash}")
    logger.info(f"Begin training with args:\n{pformat(vars(args))}")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader() if args.use_val_set else None
    with add_start_stop_flags(save_dir / tb_logger.name):
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training complete.")
    trainer.test(ckpt_path=args.which_ckpt, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="censoring")
    parser.add_argument("--seed", type=int, default=0)
    parser = Trainer.add_argparse_args(parser)
    parser = CensoredModel.add_model_specific_args(parser)
    parser = THUDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    args.git_hash = get_git_hash()
    args.which_ckpt = "best" if args.use_val_set is True else "last"
    main(args)
