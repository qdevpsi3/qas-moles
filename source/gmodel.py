import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


# Simple MLP Generator
class MLPGenerator(nn.Module):
    def __init__(self, noise_dim=16, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch_size):
        z = torch.randn(batch_size, 16, device=next(self.parameters()).device)
        return self.fc(z)


# Simple MLP Discriminator
class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [batch_size, 2]
        return self.fc(x)


# Simple MLP Predictor (Reward model)
# This predictor tries to guess y from X
class MLPPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Output is a logit for y=1
        return self.fc(x)


class GaussianGAN(LightningModule):
    def __init__(
        self,
        dataset,
        generator=None,
        discriminator=None,
        predictor=None,
        optimizer_class=torch.optim.Adam,
        lr=1e-3,
        n_critic=5,
        grad_penalty=10.0,
        train_predictor_on_fake=False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(
            ignore=["dataset", "generator", "discriminator", "predictor"]
        )

        self.dataset = dataset
        self.n_critic = n_critic
        self.grad_penalty = grad_penalty
        self.train_predictor_on_fake = train_predictor_on_fake

        self.generator = generator if generator is not None else MLPGenerator()
        self.discriminator = (
            discriminator if discriminator is not None else MLPDiscriminator()
        )
        self.predictor = predictor if predictor is not None else MLPPredictor()

        self.optimizer_class = optimizer_class
        self.lr = lr

    def configure_optimizers(self):
        g_optim = self.optimizer_class(self.generator.parameters(), lr=self.lr)
        d_optim = self.optimizer_class(self.discriminator.parameters(), lr=self.lr)
        p_optim = self.optimizer_class(self.predictor.parameters(), lr=self.lr)
        return [g_optim, d_optim, p_optim], []

    def _generate_fake_data(self, batch_size):
        x_fake = self.generator(batch_size)
        return x_fake

    def _apply_discriminator(self, x):
        return self.discriminator(x)

    def _apply_predictor(self, x):
        # Predictor outputs logits for y=1
        return self.predictor(x)

    def _calculate_gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size(), device=self.device)
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

    def _compute_discriminator_loss(self, batch):
        x_real = batch.features["X"].to(self.device)
        batch_size = x_real.size(0)

        x_fake = self._generate_fake_data(batch_size).detach()

        d_fake = self._apply_discriminator(x_fake)
        d_real = self._apply_discriminator(x_real)
        d_loss = d_fake.mean() - d_real.mean()

        # Gradient penalty
        eps = torch.rand(batch_size, 1).to(self.device)
        x_inter = (eps * x_fake + (1.0 - eps) * x_real).requires_grad_(True)
        d_inter = self._apply_discriminator(x_inter)
        grad_penalty = self._calculate_gradient_penalty(d_inter, x_inter)

        d_loss += self.grad_penalty * grad_penalty
        return d_loss

    def _compute_generator_loss(self, batch):
        batch_size = batch.features["X"].size(0)
        x_fake = self._generate_fake_data(batch_size)
        d_fake = self._apply_discriminator(x_fake)
        gan_loss = -d_fake.mean()
        rl_loss = -self._apply_predictor(x_fake).mean()
        g_loss = gan_loss + rl_loss
        return g_loss

    def _compute_predictor_loss(self, batch):
        x_real = batch.features["X"].to(self.device)
        v_real = batch.metrics["y"].to(self.device).float()  # y is either 0 or 1
        v_pred_real = self._apply_predictor(x_real)
        p_loss = nn.HuberLoss()(v_real, v_pred_real)
        aux = {"pred_real": v_pred_real.mean()}
        return p_loss, aux

    def training_step(self, batch, batch_idx):
        g_optim, d_optim, p_optim = self.optimizers()

        # Train Discriminator
        d_loss = self._compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_optim.step()
        d_optim.zero_grad()
        self.log(
            "disc_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Train Predictor
        p_loss, p_aux = self._compute_predictor_loss(batch)
        self.manual_backward(p_loss)
        p_optim.step()
        p_optim.zero_grad()
        self.log(
            "pred_loss", p_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in p_aux.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Train Generator every n_critic steps
        if (batch_idx % self.n_critic) == 0:
            g_loss = self._compute_generator_loss(batch)
            self.manual_backward(g_loss)
            g_optim.step()
            g_optim.zero_grad()
            self.log(
                "gen_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch, batch_idx):
        # Compute predictor accuracy on real data for validation
        x_real = batch.features["X"].to(self.device)
        y_real = batch.metrics["y"].to(self.device).float()
        pred_real = self._apply_predictor(x_real)
        val_acc = ((pred_real > 0).float() == y_real).float().mean()
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        # Compute predictor accuracy on test data
        x_real = batch.features["X"].to(self.device)
        y_real = batch.metrics["y"].to(self.device).float()
        pred_real = self._apply_predictor(x_real)
        test_acc = ((pred_real > 0).float() == y_real).float().mean()
        self.log("test_acc", test_acc, on_epoch=True, prog_bar=True, logger=True)
        return test_acc
