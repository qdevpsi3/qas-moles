import gzip
import math
import pickle
import warnings
from functools import partial

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen
from torchmetrics import Metric

# Suppress RDKit warnings using RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class MolecularMetric(Metric):
    """Base class for calculating molecular metrics. Implements basic functionality for metric calculation and aggregation."""

    def __init__(self, dist_sync_on_step=False):
        """Initializes state for metric computation."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _valid_molecule(mol):
        """Checks if a molecule is valid and can be converted to a SMILES string."""
        return mol is not None and Chem.MolToSmiles(mol) != ""

    def update(self, mols):
        """Updates the metric's state with the provided molecules."""
        scores = self.compute_score(mols)
        scores = np.array(scores, dtype=np.float32)
        self.score += torch.tensor(scores).sum()
        self.total += len(mols)

    def compute(self):
        """Computes the final metric score."""
        return self.score.float() / self.total.float()

    def compute_score(self, mols):
        """Method to compute the score of the given molecules. Should be implemented by subclasses."""
        raise NotImplementedError("This method needs to be implemented by subclasses")


def _avoid_sanitization_error(op):
    """Wrapper function to handle RDKit sanitization errors gracefully."""
    try:
        return op()
    except ValueError:
        return None


def _map_score_func(func, default: float):
    """Maps a scoring function over a list of molecules with error handling."""

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
    """Normalizes scores to be between 0 and 1."""
    return (scores - score_range[0]) / (score_range[1] - score_range[0])


def _raise_norm_warning(norm):
    """Raises a warning if normalization is specified but not used."""
    if norm is not None:
        warnings.warn(
            "Norm was specified but will not used, as in MolGan implementation."
        )


def _constant_bump(x, x_low, x_high, decay=0.025):
    """Applies a bump function to values, primarily for normalization and scaling."""
    return np.select(
        condlist=[x <= x_low, x >= x_high],
        choicelist=[
            np.exp(-((x - x_low) ** 2) / decay),
            np.exp(-((x - x_high) ** 2) / decay),
        ],
        default=np.ones_like(x),
    )


class WOPCScore(MolecularMetric):
    """Calculates the water-octanol partition coefficient (LogP) of molecules."""

    _molgan_label = "logp"
    _molgan_func = "water_octanol_partition_coefficient"

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
    """Calculates the Quantitative Estimate of Drug-likeness (QED) of molecules."""

    _molgan_label = "qed"
    _molgan_func = "quantitative_estimation_druglikeness"

    def __init__(self, norm=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        _raise_norm_warning(norm)
        self.norm = norm

    def compute_score(self, mols):
        scores = _map_score_func(QED.qed, 0.0)(mols)
        return scores


class NPScore(MolecularMetric):
    """Calculates a score based on the likeness to natural products."""

    _molgan_label = "np"
    _molgan_func = "natural_product"

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

        # Calculates the score based on a model of natural product likeness
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

        # Logarithmic transformation for scores outside normal range
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
    """Calculates the synthetic accessibility score of molecules."""

    _molgan_label = "sas"
    _molgan_func = "synthetic_accessibility"

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
    """Calculates the novelty of molecules compared to a given dataset."""

    _molgan_label = "novelty"
    _molgan_func = "novel"

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
    """Combines multiple molecular metrics to evaluate drug candidacy."""

    _molgan_label = "dc"
    _molgan_func = "drugcandidate"

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
    """Calculates the uniqueness of molecules within a batch."""

    _molgan_label = "unique"
    _molgan_func = "unique"

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


class DiversityScore(MolecularMetric):
    """Calculates the chemical diversity of molecules within a batch relative to a dataset."""

    _molgan_label = "diversity"
    _molgan_func = "diversity"

    def __init__(self, data, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.data = data

    def compute_score(self, mols):
        def _compute_diversity(mol, fps):
            ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 4, nBits=2048
            )
            dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
            score = np.mean(dist)
            return score

        rand_mols = np.random.choice(self.data.data, 100)
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            for mol in rand_mols
        ]
        scores = _map_score_func(partial(_compute_diversity, fps=fps), 0.0)(mols)
        scores = _normalize_score(scores, (0.9, 0.945))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class ValidityScore(MolecularMetric):
    """Checks the chemical validity of molecules, ensuring they are properly formed without disjoint fragments."""

    _molgan_label = "validity"
    _molgan_func = "valid"

    def compute_score(self, mols):
        def _compute_valid(mol):
            s = Chem.MolToSmiles(mol) if mol is not None else ""
            score = "*" not in s and "." not in s and s != ""
            return float(score)

        scores = _map_score_func(_compute_valid, 0.0)(mols)
        return scores


ALL_METRICS = {
    "logp": WOPCScore,
    "qed": QEDScore,
    "np": NPScore,
    "sas": SASScore,
    "novelty": NoveltyScore,
    "dc": DCScore,
    "unique": UniqueScore,
    "diversity": DiversityScore,
    "validity": ValidityScore,
}

if __name__ == "__main__":
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC"), None]
    for metric in [
        WOPCScore(),
        QEDScore(),
        NPScore(),
        SASScore(),
        UniqueScore(),
        ValidityScore(),
    ]:
        metric.update(mols)
        score = metric.compute()
        scores = metric.compute_score(mols)
        print(metric._molgan_label, score, scores)
