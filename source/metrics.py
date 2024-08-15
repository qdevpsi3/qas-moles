import gzip
import math
import pickle
import warnings

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
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


def _raise_norm_warning(norm):
    if norm is not None:
        warnings.warn(
            "Norm was specified but will not used, as in MolGan implementation."
        )


def _constant_bump(x, x_low, x_high, decay=0.025):
    return np.select(
        condlist=[x <= x_low, x >= x_high],
        choicelist=[
            np.exp(-((x - x_low) ** 2) / decay),
            np.exp(-((x - x_high) ** 2) / decay),
        ],
        default=np.ones_like(x),
    )


class WOPCScore(MolecularMetric):
    _molgan_label = "logp"
    _molgan_long_name = "water_octanol_partition_coefficient"

    def __init__(self, norm=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.norm = norm

    def compute_score(self, mols):
        scores = _map_score_func(Crippen.MolLogP, -3.0)(mols)
        if self.norm:
            scores = _normalize_score(scores, (-2.12178879609, 6.0429063424))
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class QEDScore(MolecularMetric):
    _molgan_label = "qed"
    _molgan_long_name = "quantitative_estimation_druglikeness"

    def __init__(self, norm=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        _raise_norm_warning(norm)
        self.norm = norm

    def compute_score(self, mols):
        scores = _map_score_func(QED.qed, 0.0)(mols)
        return scores


class NPScore(MolecularMetric):
    _molgan_label = "np"
    _molgan_long_name = "natural_product"

    def __init__(
        self,
        norm=False,
        np_model_path="./data/NP_score.pkl.gz",
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.norm = norm
        self.np_model_path = np_model_path
        self.np_model = pickle.load(gzip.open(np_model_path))

    def compute_score(self, mols):

        # calculating the score
        scores = [
            (
                sum(
                    self.np_model.get(bit, 0)
                    for bit in Chem.rdMolDescriptors.GetMorganFingerprint(
                        mol, 2
                    ).GetNonzeroElements()
                )
                / float(mol.GetNumAtoms())
                if mol is not None
                else None
            )
            for mol in mols
        ]

        # preventing score explosion for exotic molecules
        scores = list(
            map(
                lambda score: (
                    score
                    if score is None
                    else (
                        4 + math.log10(score - 4 + 1)
                        if score > 4
                        else (-4 - math.log10(-4 - score + 1) if score < -4 else score)
                    )
                ),
                scores,
            )
        )

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        if self.norm:
            scores = _normalize_score(scores, (-3.0, 1.0))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class SASScore(MolecularMetric):
    _molgan_label = "sas"
    _molgan_long_name = "synthetic_accessibility"

    def __init__(
        self,
        norm=False,
        sa_model_path="./data/SA_score.pkl.gz",
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.norm = norm
        self.sa_model_path = sa_model_path
        self.sa_model = {
            i[j]: float(i[0])
            for i in pickle.load(gzip.open(self.sa_model_path))
            for j in range(1, len(i))
        }

    def _compute_SAS(self, mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0

        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += self.sa_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = (
            0.0
            - sizePenalty
            - stereoPenalty
            - spiroPenalty
            - bridgePenalty
            - macrocyclePenalty
        )

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.0
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        _min = -4.0
        _max = 2.5
        sascore = 11.0 - (sascore - _min + 1) / (_max - _min) * 9.0
        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    def compute_score(self, mols):
        scores = _map_score_func(self._compute_SAS, 10.0)(mols)
        if self.norm:
            scores = _normalize_score(scores, (5, 1.5))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class NoveltyScore(MolecularMetric):
    _molgan_label = "novelty"
    _molgan_long_name = "novel"

    def __init__(self, data, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.data = data

    def compute_score(self, mols):
        def _score_func(mol):
            cond_1 = Chem.MolToSmiles(mol) != ""
            cond_2 = Chem.MolToSmiles(mol) not in self.data.smiles
            return float(cond_1 and cond_2)

        scores = _map_score_func(_score_func, 0.0)(mols)
        return scores


class DCScore(MolecularMetric):
    _molgan_label = "dc"
    _molgan_long_name = "drugcandidate"

    def __init__(self, data, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.data = data

    def compute_score(self, mols):
        scores = 0.0
        scores += _constant_bump(WOPCScore(norm=True)(mols), 0.210, 0.945)
        scores += SASScore(norm=True)(mols)
        scores += NoveltyScore(self.data)(mols)
        scores += (1 - NoveltyScore(self.data)(mols)) * 0.3
        scores /= 4.0
        return scores


class UniqueScore(MolecularMetric):
    _molgan_label = "unique"
    _molgan_long_name = "unique"

    def compute_score(self, mols):
        smiles = list(
            map(
                lambda x: (Chem.MolToSmiles(x) if x is not None else ""),
                mols,
            )
        )
        scores = 0.75 + np.array(
            list(map(lambda x: 1 / smiles.count(x) if x != "" else 0, smiles)),
            dtype=np.float32,
        )
        scores = np.clip(scores, 0.0, 1.0)
        return scores


if __name__ == "__main__":
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC"), None]
    for metric in [WOPCScore(), QEDScore(), NPScore(), SASScore(), UniqueScore()]:
        metric.update(mols)
        score = metric.compute()
        scores = metric.compute_score(mols)
        print(metric._molgan_label, score, scores)
