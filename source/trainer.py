import torch
import torch.nn.functional as F
from lightning import LightningModule


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
        generator,
        discriminator,
        predictor,
        optimizer,
        *,
        grad_penalty=10.0,
        process_method="soft_gumbel",
        m_dim=5,
        b_dim=5,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["generator", "discriminator", "predictor", "optimizer"],
        )
        self.generator = generator
        self.discriminator = discriminator
        self.predictor = predictor
        self.optimizer = optimizer

    def configure_optimizers(self):
        g_optim = self.optimizer(self.generator.parameters())
        d_optim = self.optimizer(self.discriminator.parameters())
        p_optim = self.optimizer(self.predictor.parameters())
        return [g_optim, d_optim, p_optim], []

    def _calculate_gradient_penalty(self, x, x_hat):
        alpha = torch.rand(x.size(0), 1, 1, 1, device=self.device)
        interpolated = alpha * x + (1 - alpha) * x_hat
        interpolated.requires_grad_(True)
        scores_on_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores_on_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores_on_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_norm = gradients.norm(2, dim=1)
        grad_penalty = ((gradient_norm - 1) ** 2).mean()
        return grad_penalty * self.hparams.grad_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        # train generator
        if optimizer_idx == 0:
            return self.generator_step(x)
        # train discriminator
        if optimizer_idx == 1:
            return self.discriminator_step(x)
        return None

    def _generate_noise(self, batch_size):
        return torch.rand(batch_size, self.hparams.z_dim, device=self.device)

    def _generate_fake_data(self, batch):
        z = self._generate_noise(batch.size(0))
        a_fake, x_fake = self.generator(z)
        return a_fake, x_fake

    def _process_real_data(self, a_real, x_real):
        a_real = label2onehot(a_real, self.hparams.b_dim)
        x_real = label2onehot(x_real, self.hparams.m_dim)
        return a_real, x_real

    def _process_fake_data(self, a_fake, x_fake):
        a_fake, x_fake = postprocess(
            [a_fake, x_fake],
            method=self.hparams.process_method,
        )
        return a_fake, x_fake

    def _apply_discriminator(self, a, x):
        return self.discriminator(a, None, x)[0]

    def _compute_discriminator_loss(self, batch):
        a_real, x_real = batch["A"], batch["X"]
        a_real, x_real = self._process_real_data(a_real, x_real)
        a_fake, x_fake = self._generate_fake_data(batch)
        a_fake, x_fake = a_fake.detach(), x_fake.detach()  # detach fake data
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        d_fake = self._apply_discriminator(a_fake, x_fake)
        d_real = self._apply_discriminator(*batch)
        grad_penalty = self._calculate_gradient_penalty(
            torch.cat([a_real, a_fake]), torch.cat([x_real, x_fake])
        )
        d_loss = d_fake.mean() - d_real.mean() + grad_penalty
        return d_loss

    def _compute_generator_loss(self, batch):
        a_fake, x_fake = self._generate_fake_data(batch)
        a_fake, x_fake = self._process_fake_data(a_fake, x_fake)
        d_fake = self._apply_discriminator(a_fake, x_fake)
        g_loss = -d_fake.mean()
        return g_loss
