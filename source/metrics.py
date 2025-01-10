import gzip
import math
import multiprocessing
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


def _call_rdkit_func(func, mol):
    """
    Safely call an RDKit function (e.g., Crippen.MolLogP) on a molecule,
    returning None if a sanitization error (ValueError) occurs.
    """
    try:
        return func(mol)
    except ValueError:
        return None


def _worker_func(args):
    """
    The function we will call in parallel.
    `args` is a tuple (mol, func, default).
    """
    mol, func, default = args
    if mol is None:
        return default

    val = _call_rdkit_func(func, mol)
    return val if val is not None else default


def _normalize_score(scores, score_range):
    """Normalizes scores to be between 0 and 1."""
    return (scores - score_range[0]) / (score_range[1] - score_range[0])


def _raise_norm_warning(norm):
    """Raises a warning if normalization is specified but not used."""
    if norm is not None:
        warnings.warn(
            "Norm was specified but will not be used, as in MolGan implementation."
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


class MolecularMetric(Metric):
    """Base class for calculating molecular metrics. Implements basic functionality for metric calculation and aggregation."""

    def __init__(self, dist_sync_on_step=False, parallel=False, n_jobs=None):
        """
        :param dist_sync_on_step: Whether to synchronize during distributed training (TorchMetrics param).
        :param parallel: If True, use multiprocessing to speed up the scoring function.
        :param n_jobs: Number of processes to use if parallel=True. If None, uses all CPU cores.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.parallel = parallel
        self.n_jobs = n_jobs

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
        """Method to compute the score of the given molecules. Must be implemented by subclasses."""
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _map_scoring_func(self, func, default, mols):
        """
        Applies the scoring function to each molecule, either sequentially or in parallel.

        :param func: Top-level function that takes (mol) -> float or None
        :param default: Default score if function fails or mol is invalid
        :param mols: List of RDKit molecules
        """
        if not self.parallel:
            # Sequential
            results = []
            for mol in mols:
                if mol is None:
                    results.append(default)
                else:
                    val = _call_rdkit_func(func, mol)
                    results.append(val if val is not None else default)
            return np.array(results, dtype=float)

        else:
            # Parallel
            processes = self.n_jobs or multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=processes) as pool:
                # Each item in the list is (mol, func, default)
                args_iter = [(mol, func, default) for mol in mols]
                results = pool.map(_worker_func, args_iter)
            return np.array(results, dtype=float)


class WOPCScore(MolecularMetric):
    """Calculates the water-octanol partition coefficient (LogP) of molecules."""

    _molgan_label = "logp"
    _molgan_func = "water_octanol_partition_coefficient"

    def __init__(self, norm=True, dist_sync_on_step=False, parallel=False, n_jobs=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        self.norm = norm

    def compute_score(self, mols):
        # Range from RDKit or your domain knowledge
        scores = self._map_scoring_func(Crippen.MolLogP, -3.0, mols)
        if self.norm:
            scores = _normalize_score(scores, (-2.12178879609, 6.0429063424))
            scores = np.clip(scores, 0.0, 1.0)
        return scores


class QEDScore(MolecularMetric):
    """Calculates the Quantitative Estimate of Drug-likeness (QED) of molecules."""

    _molgan_label = "qed"
    _molgan_func = "quantitative_estimation_druglikeness"

    def __init__(self, norm=None, dist_sync_on_step=False, parallel=False, n_jobs=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        _raise_norm_warning(norm)
        self.norm = norm

    def compute_score(self, mols):
        # If norm is relevant, handle it here
        scores = self._map_scoring_func(QED.qed, 0.0, mols)
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
        parallel=False,
        n_jobs=None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        self.norm = norm
        self.np_model_path = np_model_path
        self.np_model = pickle.load(gzip.open(np_model_path, "rb"))

    def compute_score(self, mols):
        raw_scores = []
        for mol in mols:
            if mol is None:
                raw_scores.append(None)
                continue

            bits = Chem.rdMolDescriptors.GetMorganFingerprint(
                mol, 2
            ).GetNonzeroElements()
            score = sum(self.np_model.get(bit, 0) for bit in bits) / float(
                mol.GetNumAtoms()
            )
            raw_scores.append(score)

        # Logarithmic transform for out-of-range scores
        transformed = []
        for sc in raw_scores:
            if sc is None:
                transformed.append(-4)
            else:
                if sc > 4:
                    sc = 4 + math.log10(sc - 4 + 1)
                elif sc < -4:
                    sc = -4 - math.log10(-4 - sc + 1)
                transformed.append(sc)

        scores = np.array(transformed, dtype=float)
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
        parallel=False,
        n_jobs=None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        self.norm = norm
        self.sa_model_path = sa_model_path
        # Flatten the SA model
        loaded_sa = pickle.load(gzip.open(self.sa_model_path, "rb"))
        self.sa_model = {i[j]: float(i[0]) for i in loaded_sa for j in range(1, len(i))}

    def _compute_SAS(self, mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0

        for bitId, v in fps.items():
            nf += v
            score1 += self.sa_model.get(bitId, -4) * v
        score1 /= max(nf, 1e-9)  # avoid division by zero

        # Additional features
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        # Macrocycles
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0
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

        # Correction for fingerprint density
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5
        else:
            score3 = 0.0

        sascore = score1 + score2 + score3

        # Transform "raw" value into scale between 1 and 10
        _min = -4.0
        _max = 2.5
        sascore = 11.0 - (sascore - _min + 1) / (_max - _min) * 9.0

        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    def compute_score(self, mols):
        # For full parallelization, you could do _map_scoring_func(self._compute_SAS, 10.0, mols)
        # Instead, we do a simple loop here:
        scores = []
        for mol in mols:
            if mol is not None:
                val = self._compute_SAS(mol)
                scores.append(val)
            else:
                scores.append(10.0)  # default for invalid
        scores = np.array(scores, dtype=float)
        if self.norm:
            scores = _normalize_score(scores, (5.0, 1.5))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class NoveltyScore(MolecularMetric):
    """Calculates the novelty of molecules compared to a given dataset."""

    _molgan_label = "novelty"
    _molgan_func = "novel"

    def __init__(self, data, dist_sync_on_step=False, parallel=False, n_jobs=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        self.data = data

    def compute_score(self, mols):
        def _score_func(mol):
            s = Chem.MolToSmiles(mol) if mol is not None else ""
            return float(s != "" and s not in self.data.smiles)

        scores = self._map_scoring_func(_score_func, 0.0, mols)
        return scores


class DCScore(MolecularMetric):
    """Combines multiple molecular metrics to evaluate drug candidacy."""

    _molgan_label = "dc"
    _molgan_func = "drugcandidate"

    def __init__(self, data, dist_sync_on_step=False, parallel=False, n_jobs=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        self.data = data

    def compute_score(self, mols):
        # Weighted sum of different metrics
        # Notice here we instantiate other metrics inline (they inherit the same parallel/n_jobs if you like)
        wopc = WOPCScore(norm=True, parallel=self.parallel, n_jobs=self.n_jobs)(mols)
        sas = SASScore(norm=True, parallel=self.parallel, n_jobs=self.n_jobs)(mols)
        novelty = NoveltyScore(self.data, parallel=self.parallel, n_jobs=self.n_jobs)(
            mols
        )

        # Example combination
        scores = 0.0
        scores += _constant_bump(wopc, 0.210, 0.945)
        scores += sas
        scores += novelty
        scores += (1 - novelty) * 0.3
        scores /= 4.0
        return scores


class UniqueScore(MolecularMetric):
    """Calculates the uniqueness of molecules within a batch."""

    _molgan_label = "unique"
    _molgan_func = "unique"

    def compute_score(self, mols):
        smiles = [Chem.MolToSmiles(x) if x is not None else "" for x in mols]
        # "Uniqueness" can be defined many ways; here we do 1 / frequency
        scores = 0.75 + np.array(
            [1 / smiles.count(s) if s != "" else 0 for s in smiles],
            dtype=np.float32,
        )
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class DiversityScore(MolecularMetric):
    """Calculates the chemical diversity of molecules within a batch relative to a dataset."""

    _molgan_label = "diversity"
    _molgan_func = "diversity"

    def __init__(self, data, dist_sync_on_step=False, parallel=False, n_jobs=None):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, parallel=parallel, n_jobs=n_jobs
        )
        self.data = data

    def compute_score(self, mols):
        def _compute_diversity(mol, fps):
            ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 4, nBits=2048
            )
            dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
            return np.mean(dist)

        rand_mols = np.random.choice(self.data.data, 100)
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            for mol in rand_mols
        ]

        # We can easily parallelize with self._map_scoring_func
        def func(mol):
            return _compute_diversity(mol, fps) if mol is not None else 0.0

        scores = self._map_scoring_func(func, 0.0, mols)
        # Just an example range
        scores = _normalize_score(scores, (0.9, 0.945))
        scores = np.clip(scores, 0.0, 1.0)
        return scores


class ValidityScore(MolecularMetric):
    """Checks the chemical validity of molecules (no '*' or '.' in SMILES, etc.)."""

    _molgan_label = "validity"
    _molgan_func = "valid"

    def compute_score(self, mols):
        def _compute_valid(mol):
            s = Chem.MolToSmiles(mol) if mol is not None else ""
            return float("*" not in s and "." not in s and s != "")

        scores = self._map_scoring_func(_compute_valid, 0.0, mols)
        return scores


# Dictionary of all metrics for quick access
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
    # Example usage:
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC"), None]

    # Suppose you want to run in parallel with 2 processes:
    metrics_to_try = [
        WOPCScore(parallel=False, n_jobs=2),
        # QEDScore(parallel=True, n_jobs=2),
        # SASScore(parallel=True, n_jobs=2),
        # UniqueScore(parallel=True, n_jobs=2),
        # ValidityScore(parallel=True, n_jobs=2),
    ]

    for metric in metrics_to_try:
        metric.update(mols)
        aggregated_score = metric.compute()  # aggregated
        per_molecule_scores = metric.compute_score(mols)  # per-molecule
        print(
            f"{metric._molgan_label} -> Aggregated: {aggregated_score}, Per-mol: {per_molecule_scores}"
        )
