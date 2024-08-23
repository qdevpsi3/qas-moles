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
        method="soft_gumbel",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
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
