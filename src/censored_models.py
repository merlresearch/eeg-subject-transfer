# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import gc
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import accuracy


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Residual(nn.Module):
    def __init__(self, layer: nn.Module):
        """NOTE - output shape must match input shape"""
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


class Projector(nn.Sequential):
    def __init__(self, dims: List[int], use_spectral_norm=False):
        final_dim = dims.pop(-1)
        layers = []

        # Add hidden layers with norm and activation
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim, bias=False)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layer = nn.Sequential(linear, nn.BatchNorm1d(out_dim))
            if in_dim == out_dim:
                layer = Residual(layer)
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))

        # Add final layer with bias and without norm or activation
        layer = nn.Linear(out_dim, final_dim)
        if use_spectral_norm:
            layer = spectral_norm(layer)
        if out_dim == final_dim:
            layer = Residual(layer)
        layers.append(layer)
        super().__init__(*layers)


class MainModel(nn.Module):
    """Given raw sensor data, output task logits and latent features."""

    def __init__(
        self, n_classes: int, input_shape: Tuple[int], latent_dim: int, feature_type: str, proj_hidden_layers: int
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, kernel_size=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.05),
            nn.MaxPool1d(kernel_size=2),
            #
            nn.Conv1d(32, 128, kernel_size=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.05),
            nn.MaxPool1d(kernel_size=2),
        )
        T = self.encoder(torch.ones(1, *input_shape))
        self.encoder.append(nn.Conv1d(128, self.latent_dim, kernel_size=T.shape[2:], bias=True))
        self.encoder.append(nn.Flatten())

        if feature_type == "direct":
            self.projector = nn.Identity()
        elif feature_type == "projected":
            dims = [self.latent_dim] + ([128] * proj_hidden_layers) + [self.latent_dim]
            self.projector = Projector(dims)
        self.classifier = Projector([self.latent_dim, 128, 128, 128, n_classes])

    def forward(self, x):
        hidden_features = self.encoder(x)
        features = self.projector(hidden_features)
        logits = self.classifier(hidden_features)
        return logits, features


class CensoredModel(LightningModule):
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--feature_type", choices=["direct", "projected"], default="projected", help="how encodings are censored"
        )
        parser.add_argument("--main_lr", type=float, default=0.003, help="main lr")
        parser.add_argument("--main_lr_decay", type=float, default=1.0, help="main lr decay")
        parser.add_argument("--censor_lr", type=float, default=0.003, help="censor lr")
        parser.add_argument("--censor_lr_decay", type=float, default=1.0, help="censor lr decay")
        parser.add_argument("--latent_dim", type=int, default=128, help="dimension of features for censoring")
        parser.add_argument("--censor_type", choices=["wyner", "wasserstein", "adv"], required=True)
        parser.add_argument(
            "--censor_mode",
            choices=["marginal", "conditional", "complementary"],
            default="marginal",
            help="marginal: z ⟂ s. conditional: z⟂s | y. complementary: z=(z1, z2). z1⟂s, max(I(z2; s))",
        )
        parser.add_argument("--censor_weight", type=float, default=1.0)
        parser.add_argument(
            "--censor_exclude_epochs",
            nargs="+",
            type=int,
            default=None,
            help="pass 1 or more ints to set censor_weight=0 for those epochs",
        )
        parser.add_argument(
            "--censor_steps_per_main_step", type=int, default=1, help="censor model steps per main model step"
        )
        parser.add_argument("--censor_hidden_layers", type=int, default=3, help="hidden layers of censor model")
        parser.add_argument(
            "--proj_hidden_layers",
            type=int,
            default=3,
            help="hidden layers of projection. only used for feature_type='projected'",
        )
        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # saves arg1 to self.hparam.arg1, etc
        self.main_model = MainModel(
            n_classes=self.hparams.num_classes,
            input_shape=self.hparams.input_shape,
            latent_dim=self.hparams.latent_dim,
            feature_type=self.hparams.feature_type,
            proj_hidden_layers=self.hparams.proj_hidden_layers,
        )

        if self.hparams.censor_type in ["adv"]:
            if self.hparams.censor_mode == "marginal":
                input_dim = self.hparams.latent_dim
            elif self.hparams.censor_mode == "conditional":
                input_dim = self.hparams.latent_dim + self.hparams.num_classes
            elif self.hparams.censor_mode == "complementary":
                assert self.hparams.latent_dim % 2 == 0
                input_dim = self.hparams.latent_dim // 2
            else:
                raise NotImplementedError()
            dims = [input_dim] + ([128] * self.hparams.censor_hidden_layers) + [self.hparams.n_train_subj]
        else:  # wyner, wasserstein
            if self.hparams.censor_mode == "marginal":
                input_dim = self.hparams.latent_dim + self.hparams.n_train_subj
            elif self.hparams.censor_mode == "conditional":
                input_dim = self.hparams.latent_dim + self.hparams.n_train_subj + self.hparams.num_classes
            elif self.hparams.censor_mode == "complementary":
                assert self.hparams.latent_dim % 2 == 0
                input_dim = self.hparams.latent_dim // 2 + self.hparams.n_train_subj
            else:
                raise NotImplementedError()
            dims = [input_dim] + ([128] * self.hparams.censor_hidden_layers) + [1]
        self.censor_model = Projector(dims, use_spectral_norm=(self.hparams.censor_type == "wasserstein"))
        self.automatic_optimization = False
        self.example_input_array = torch.randn(1, *self.hparams.input_shape)

    def setup(self, stage: Optional[str] = None) -> None:
        device = self.trainer.strategy.root_device.type  # https://github.com/Lightning-AI/lightning/issues/13108
        self.class_weights = torch.tensor(self.hparams.class_weights, requires_grad=False).to(device)

    def forward(self, x):
        logits, features = self.main_model(x)
        return logits, features

    # Censoring loss functions
    def wyner_criterion(self, features, labels, subj_ids, subj_ids_shuf, train_which: str = "main_model"):
        if self.hparams.censor_mode == "marginal":
            joint = self.censor_model(torch.cat((features, subj_ids), dim=-1))
            marginal = self.censor_model(torch.cat((features, subj_ids_shuf), dim=-1))
        elif self.hparams.censor_mode == "conditional":
            # NOTE: we want to minimize the mutual information between Z and S, conditioned on Y
            # We can express this with either of 2 conditional MI terms:
            # I(Z ; S | Y) or I(S ; Z | Y). We choose the second one for convenience.
            # Chain rule of mutual information says: I(S; Z | Y) + I(Y; S) = I(Z, Y; S)
            # Observe that I(Y; S) is constant during our optimization.
            # Thus we can minimize I(Z, Y; S) in order to minimize I(S; Z | Y)
            # To minimize I(Z, Y; S), we compare real samples of (Z, Y, S) with samples of (Z, Y, shuffled_S)
            joint = self.censor_model(torch.cat((features, labels, subj_ids), dim=-1))
            marginal = self.censor_model(torch.cat((features, labels, subj_ids_shuf), dim=-1))
        elif self.hparams.censor_mode == "complementary":
            dim = features.shape[1] // 2
            z1 = features[:, :dim]
            z2 = features[:, dim:]
            joint1 = self.censor_model(torch.cat((z1, subj_ids), dim=-1))
            marginal1 = self.censor_model(torch.cat((z1, subj_ids_shuf), dim=-1))
            joint2 = self.censor_model(torch.cat((z2, subj_ids), dim=-1))
            marginal2 = self.censor_model(torch.cat((z2, subj_ids_shuf), dim=-1))
            if train_which == "censor_model":
                term1 = -F.logsigmoid(joint1).mean() - F.logsigmoid(-marginal1).mean()
                term2 = -F.logsigmoid(joint2).mean() - F.logsigmoid(-marginal2).mean()
                return term1 - term2
            elif train_which == "main_model":
                term1 = joint1.mean()
                term2 = joint2.mean()
                return term1 - term2

        else:
            raise NotImplementedError()

        if train_which == "censor_model":
            return -F.logsigmoid(joint).mean() - F.logsigmoid(-marginal).mean()
        elif train_which == "main_model":
            return joint.mean()
        else:
            raise NotImplementedError()

    def wasserstein_criterion(self, features, labels, subj_ids, subj_ids_shuf, train_which: str = "main_model"):
        if self.hparams.censor_mode == "marginal":
            joint = self.censor_model(torch.cat((features, subj_ids), dim=-1))
            marginal = self.censor_model(torch.cat((features, subj_ids_shuf), dim=-1))
            dependence_est = joint.mean() - marginal.mean()
        elif self.hparams.censor_mode == "conditional":
            # See note in wyner_criterion above
            joint = self.censor_model(torch.cat((features, labels, subj_ids), dim=-1))
            marginal = self.censor_model(torch.cat((features, labels, subj_ids_shuf), dim=-1))
            dependence_est = joint.mean() - marginal.mean()
        elif self.hparams.censor_mode == "complementary":
            dim = features.shape[1] // 2
            z1 = features[:, :dim]
            z2 = features[:, dim:]
            joint1 = self.censor_model(torch.cat((z1, subj_ids), dim=-1))
            marginal1 = self.censor_model(torch.cat((z1, subj_ids_shuf), dim=-1))
            joint2 = self.censor_model(torch.cat((z2, subj_ids), dim=-1))
            marginal2 = self.censor_model(torch.cat((z2, subj_ids_shuf), dim=-1))
            term1 = joint1.mean() - marginal1.mean()
            term2 = joint2.mean() - marginal2.mean()
            dependence_est = term1 - term2
        else:
            raise NotImplementedError()

        if train_which == "main_model":
            return -dependence_est
        elif train_which == "censor_model":
            return dependence_est
        else:
            raise NotImplementedError()

    def adv_criterion(self, features, labels, subj_ids, train_which: str = "main_model"):
        if self.hparams.censor_mode == "marginal":
            logits = self.censor_model(features)
            cross_ent = F.cross_entropy(logits, subj_ids)
        elif self.hparams.censor_mode == "conditional":
            logits = self.censor_model(torch.cat((features, labels), dim=-1))
            cross_ent = F.cross_entropy(logits, subj_ids)
        elif self.hparams.censor_mode == "complementary":
            dim = features.shape[1] // 2
            z1 = features[:, :dim]
            z2 = features[:, dim:]
            logits1 = self.censor_model(z1)
            logits2 = self.censor_model(z2)
            term1 = F.cross_entropy(logits1, subj_ids)
            term2 = F.cross_entropy(logits2, subj_ids)
            cross_ent = term1 - term2
        else:
            raise NotImplementedError()

        if train_which == "main_model":
            return -cross_ent
        elif train_which == "censor_model":
            return cross_ent
        else:
            raise NotImplementedError()

    def on_train_start(self):
        logger.info(f"Main model layers:\n{self.main_model}")
        logger.info(f"Censoring layers:\n{self.censor_model}")
        logger.info(f"Encoder params: {count_params(self.main_model.encoder)}")
        logger.info(f"Projector params: {count_params(self.main_model.projector)}")
        logger.info(f"Classifier params: {count_params(self.main_model.classifier)}")
        logger.info(f"Censoring params: {count_params(self.censor_model)}")

    def get_censor_weight(self):
        exclude_epochs = self.hparams.censor_exclude_epochs
        if exclude_epochs is not None and self.current_epoch in exclude_epochs:
            return 0.0
        else:
            return self.hparams.censor_weight

    def get_censor_loss(self, features, labels, subj_ids, train_which: str):
        labels = F.one_hot(labels, num_classes=self.hparams.num_classes).to(labels.device)
        if self.hparams.censor_type in ["wyner", "wasserstein"]:
            # One-hot version of labels, and shuffled one-hot
            subj_ids = F.one_hot(subj_ids, num_classes=self.hparams.n_train_subj).to(subj_ids.device)
            subj_ids_shuf = subj_ids[torch.randperm(len(subj_ids))]

            if self.hparams.censor_type == "wyner":
                return self.wyner_criterion(features, labels, subj_ids, subj_ids_shuf, train_which)
            elif self.hparams.censor_type == "wasserstein":
                return self.wasserstein_criterion(features, labels, subj_ids, subj_ids_shuf, train_which)
            else:
                raise ValueError()
        elif self.hparams.censor_type == "adv":
            return self.adv_criterion(features, labels, subj_ids, train_which)
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        main_opt, censor_opt = self.optimizers()
        data, labels, subj_ids = batch

        # Train censoring model
        for _ in range(self.hparams.censor_steps_per_main_step):
            logits, features = self.forward(data)
            censor_loss = self.get_censor_loss(features.detach(), labels, subj_ids, train_which="censor_model")
            censor_opt.zero_grad()
            self.manual_backward(censor_loss)
            censor_opt.step()

        # Train main model
        logits, features = self.forward(data)
        cross_ent = F.cross_entropy(logits, labels, weight=self.class_weights)
        censor_loss = self.get_censor_loss(features, labels, subj_ids, train_which="main_model")
        censor_weight = self.get_censor_weight()
        main_loss = cross_ent + censor_weight * censor_loss
        main_opt.zero_grad()
        self.manual_backward(main_loss)
        main_opt.step()

        bal_acc = accuracy(
            logits.argmax(-1), labels, task="multiclass", average="macro", num_classes=self.hparams.num_classes
        ).item()
        acc = logits.argmax(-1).eq(labels).float().mean().item()
        self.log("train/censor_loss", censor_loss.item())  # NOTE - we just record the final censor loss
        self.log("train/censor_weight", censor_weight)
        self.log("train/main_loss", main_loss.item())
        self.log("train/cross_ent", cross_ent.item())
        self.log("train/bal_acc", bal_acc)
        self.log("train/acc", acc)

        return None

    def training_epoch_end(self, outputs) -> None:
        main_sched, censor_sched = self.lr_schedulers()
        main_sched.step()
        censor_sched.step()
        metrics = {
            k: round(v.item(), 3) if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()
        }
        logger.info(f"Metrics: {metrics}")
        print()
        gc.collect()

    def _val_or_test(self, name, batch, batch_idx):
        data, labels, _subj_ids = batch
        logits, _features = self.main_model(data)
        cross_ent = F.cross_entropy(logits, labels, weight=self.class_weights)
        bal_acc = accuracy(
            logits.argmax(-1), labels, task="multiclass", average="macro", num_classes=self.hparams.num_classes
        ).item()
        acc = logits.argmax(-1).eq(labels).float().mean().item()
        metrics = {
            f"{name}/cross_ent": cross_ent.item(),
            f"{name}/bal_acc": bal_acc,
            f"{name}/acc": acc,
        }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self._val_or_test("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._val_or_test("test", batch, batch_idx)

    def configure_optimizers(self):
        main_optim = AdamW(self.main_model.parameters(), lr=self.hparams.main_lr)
        main_sched = ExponentialLR(main_optim, gamma=self.hparams.main_lr_decay)
        censor_optim = AdamW(self.censor_model.parameters(), lr=self.hparams.main_lr)
        censor_sched = ExponentialLR(censor_optim, gamma=self.hparams.censor_lr_decay)
        main_conf = {"optimizer": main_optim, "lr_scheduler": {"scheduler": main_sched, "name": "main_model_lr"}}
        censor_conf = {
            "optimizer": censor_optim,
            "lr_scheduler": {"scheduler": censor_sched, "name": "censor_model_lr"},
        }
        return main_conf, censor_conf
