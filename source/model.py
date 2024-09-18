import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from .metrics import ALL_METRICS


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors"""
    labels = labels.long()
    out = torch.zeros(list(labels.size()) + [dim])
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
    return out


def postprocess(inputs, method, temperature=1.0):
    """Convert the probability matrices into label matrices"""

    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == "soft_gumbel":
        softmax = [
            F.gumbel_softmax(
                e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                hard=False,
            ).view(e_logits.size())
            for e_logits in listify(inputs)
        ]
    elif method == "hard_gumbel":
        softmax = [
            F.gumbel_softmax(
                e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                hard=True,
            ).view(e_logits.size())
            for e_logits in listify(inputs)
        ]
    else:
        softmax = [
            F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)
        ]

    return [delistify(e) for e in (softmax)]


class MolGAN(LightningModule):
    def __init__(
        self,
        dataset,
        generator,
        discriminator,
        predictor,
        optimizer,
        *,
        grad_penalty=10.0,
        process_method="soft_gumbel",
        agg_method="prod",
        metrics=None,
    ):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        if metrics is None:
            metrics = ["logp", "sas", "qed", "unique"]
        self.save_hyperparameters(
            ignore=[
                "dataset",
                "generator",
                "discriminator",
                "predictor",
                "optimizer",
                "metrics",
            ],
        )
        self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor
        self.optimizer = optimizer
        self.metrics = metrics

        self.metrics_fn = dict((metric, ALL_METRICS[metric]()) for metric in metrics)

    def configure_optimizers(self):
        g_optim = self.optimizer(self.generator.parameters())
        d_optim = self.optimizer(self.discriminator.parameters())
        p_optim = self.optimizer(self.predictor.parameters())
        return [g_optim, d_optim, p_optim], []

    def _calculate_gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def training_step(self, batch, batch_idx):
        # Access the optimizers
        g_optim, d_optim, p_optim = self.optimizers()

        # train generator
        g_loss = self._compute_generator_loss(batch)
        self.manual_backward(g_loss)
        g_optim.step()
        g_optim.zero_grad()

        # train discriminator
        d_loss = self._compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_optim.step()
        d_optim.zero_grad()

        # train predictor
        p_loss, p_aux = self._compute_predictor_loss(batch)
        self.manual_backward(p_loss)
        p_optim.step()
        p_optim.zero_grad()

        # Log losses to Weights & Biases
        self.log(
            "gen_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.features["X"].size(0),
        )
        self.log(
            "disc_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.features["X"].size(0),
        )
        self.log(
            "pred_loss",
            p_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.features["X"].size(0),
        )
        for key, value in p_aux.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.features["X"].size(0),
            )
        return {"g_loss": g_loss, "d_loss": d_loss, "p_loss": p_loss}

    def _generate_fake_data(self, batch):
        batch_size = batch.features["X"].size(0)
        a_fake, x_fake = self.generator(batch_size)
        return a_fake, x_fake

    def _process_real_data(self, a_real, x_real):
        a_real = label2onehot(a_real, self.get_b_dim())
        x_real = label2onehot(x_real, self.get_m_dim())
        return a_real, x_real

    def _process_fake_data(self, a_fake, x_fake):
        a_fake, x_fake = postprocess(
            [a_fake, x_fake],
            method=self.hparams.process_method,
        )
        return a_fake, x_fake

    def _apply_discriminator(self, a, x):
        return self.discriminator(a, None, x)[0]

    def _apply_predictor(self, a, x):
        return torch.sigmoid(self.predictor(a, None, x)[0])

    def _aggregate_metrics(self, metrics):
        values = np.stack(
            [v for metric, v in metrics.items() if metric in self.metrics], axis=-1
        )
        if self.hparams.agg_method == "prod":
            return np.prod(values, axis=-1)
        elif self.hparams.agg_method == "mean":
            return np.mean(values, axis=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.hparams.agg_method}")

    def _compute_discriminator_loss(self, batch):
        a_real, x_real = batch.features["A"], batch.features["X"]
        a_real, x_real = self._process_real_data(a_real, x_real)
        a_fake, x_fake = self._generate_fake_data(batch)
        a_fake, x_fake = a_fake.detach(), x_fake.detach()  # detach fake data
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        d_fake = self._apply_discriminator(a_fake, x_fake)
        d_real = self._apply_discriminator(a_real, x_real)
        d_loss = d_fake.mean() - d_real.mean()
        # Compute gradient penalty
        eps = torch.rand(a_real.size(0), 1, 1, 1).to(self.device)
        a_inter = (eps * a_fake + (1.0 - eps) * a_real).requires_grad_(True)
        x_inter = (
            eps.squeeze(-1) * x_fake + (1.0 - eps.squeeze(-1)) * x_real
        ).requires_grad_(True)
        d_inter = self._apply_discriminator(a_inter, x_inter)
        x_penalty = self._calculate_gradient_penalty(d_inter, x_inter)
        a_penalty = self._calculate_gradient_penalty(d_inter, a_inter)
        grad_penalty = x_penalty + a_penalty
        d_loss += self.hparams.grad_penalty * grad_penalty
        return d_loss

    def _compute_generator_loss(self, batch):
        a_fake, x_fake = self._generate_fake_data(batch)
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        d_fake = self._apply_discriminator(a_fake, x_fake)
        gan_loss = -d_fake.mean()
        rl_loss = -self._apply_predictor(a_fake, x_fake).mean()
        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        a_real, x_real = batch.features["A"], batch.features["X"]
        a_real, x_real = self._process_real_data(a_real, x_real)
        metrics_real = self._compute_metrics(a_real, x_real)
        v_real = self._aggregate_metrics(metrics_real)
        a_fake, x_fake = self._generate_fake_data(batch)
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        v_fake = self._apply_predictor(a_fake, x_fake)
        metrics_fake = self._compute_metrics(a_fake, x_fake)
        v_fake = self._aggregate_metrics(metrics_fake)
        v_pred_real = self._apply_predictor(a_real, x_real)[..., 0]
        v_pred_fake = self._apply_predictor(a_fake, x_fake)[..., 0]
        v_real = torch.from_numpy(v_real).to(self.device).float()
        v_fake = torch.from_numpy(v_fake).to(self.device).float()
        p_loss_real = nn.HuberLoss()(v_real, v_pred_real)
        p_loss_fake = nn.HuberLoss()(v_fake, v_pred_fake)
        p_loss_real_per_metric = {
            metric: nn.HuberLoss()(v_real, torch.from_numpy(v).to(self.device).float())
            for metric, v in metrics_real.items()
        }
        p_loss_fake_per_metric = {
            metric: nn.HuberLoss()(v_fake, torch.from_numpy(v).to(self.device).float())
            for metric, v in metrics_fake.items()
        }
        aux = {
            metric + "/real": loss for metric, loss in p_loss_real_per_metric.items()
        }
        aux.update(
            {metric + "/fake": loss for metric, loss in p_loss_fake_per_metric.items()}
        )
        p_loss = p_loss_real + p_loss_fake
        return p_loss, aux

    def _convert_to_molecules(self, a, x):
        a, x = torch.max(a, -1)[1], torch.max(x, -1)[1]
        a, x = a.cpu().numpy(), x.cpu().numpy()
        mols = [self.dataset.matrices2mol(_x, _a, strict=True) for _a, _x in zip(a, x)]
        return mols

    def _compute_metrics(self, a, x):
        mols = self._convert_to_molecules(a, x)
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = ALL_METRICS[metric]().compute_score(mols)
        return metrics

    def get_m_dim(self):
        return self.dataset.atom_num_types

    def get_b_dim(self):
        return self.dataset.bond_num_types
