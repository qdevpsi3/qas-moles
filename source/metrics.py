import numpy as np
import torch
from rdkit import Chem
from torchmetrics import Metric


class MolecularMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _valid_molecule(mol):
        return mol is not None and Chem.MolToSmiles(mol) != ""

    def update(self, mols):
        scores = self.compute_score(mols)
        self.score += torch.tensor(scores).sum()
        self.total += len(mols)

    def compute(self):
        return self.score.float() / self.total.float()

    def compute_score(self, mols):
        raise NotImplementedError("This method needs to be implemented by subclasses")


def _avoid_sanitization_error(op):
    try:
        return op()
    except ValueError:
        return None


def _map_score_func(func, default: float):

    def wrapped_func(mols):
        return np.array(
            [
                (
                    _avoid_sanitization_error(lambda: func(mol))
                    if mol is not None
                    else default
                )
                for mol in mols
            ]
        )

    return wrapped_func


def _normalize_score(scores, score_range):
    return (scores - score_range[0]) / (score_range[1] - score_range[0])
