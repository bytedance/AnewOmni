import ray
import numpy as np

import utils.register as R

from data.bioparse.tools import load_cplx
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex

from .base import BaseFilter, FilterInput, FilterResult


def get_molecule(cplx, chain_id):
    for mol in cplx:
        if mol.id == chain_id:
            return mol
    raise KeyError(f"chain {chain_id!r} not found")


def target_ca_dict(cplx, chain_ids):
    """提取目标蛋白的 CA，用于候选结构与参考结构对齐。"""
    coords = {}

    for mol in cplx:
        if mol.id not in chain_ids:
            continue

        for block in mol:
            for atom in block:
                if atom.name == "CA":
                    key = (mol.id, block.id)
                    coords[key] = np.asarray(
                        atom.get_coord(),
                        dtype=np.float64,
                    )

    return coords


def motif_coords(
    cplx,
    chain_id,
    positions,
    atom_names,
    expected_residues=None,
):
    """
    positions 为肽链中的零起始位置，而不是 PDB residue number。
    """
    mol = get_molecule(cplx, chain_id)
    output = []

    for anchor_i, position in enumerate(positions):
        if position < 0 or position >= len(mol):
            raise IndexError(
                f"motif position {position} outside chain "
                f"{chain_id} length {len(mol)}"
            )

        block = mol[position]

        if expected_residues is not None:
            expected = expected_residues[anchor_i]
            if block.name != expected:
                raise ValueError(
                    f"chain {chain_id}, position {position}: "
                    f"expected {expected}, got {block.name}"
                )

        atoms = {
            atom.name: np.asarray(
                atom.get_coord(),
                dtype=np.float64,
            )
            for atom in block
        }

        for atom_name in atom_names:
            if atom_name not in atoms:
                raise ValueError(
                    f"atom {atom_name} missing from "
                    f"{chain_id}:{block.id}"
                )
            output.append(atoms[atom_name])

    return np.stack(output)


def align_to_reference(mobile, reference):
    """用 Kabsch 算法将 mobile 对齐到 reference。"""
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)

    mobile_zero = mobile - mobile_center
    reference_zero = reference - reference_center

    covariance = mobile_zero.T @ reference_zero
    u, _, vt = np.linalg.svd(covariance)

    rotation = vt.T @ u.T

    # 防止出现镜像变换
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = reference_center - rotation @ mobile_center
    return rotation, translation


@R.register("MotifBackboneRMSDFilter")
class MotifBackboneRMSDFilter(BaseFilter):

    def __init__(
        self,
        reference_path,
        ligand_chain,
        anchor_positions,
        cutoff=0.75,
        atom_names=None,
        expected_residues=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reference_path = reference_path
        self.ligand_chain = ligand_chain
        self.anchor_positions = anchor_positions
        self.cutoff = float(cutoff)

        self.atom_names = (
            ["N", "CA", "C", "O"]
            if atom_names is None
            else atom_names
        )

        self.expected_residues = expected_residues

    @property
    def name(self):
        return (
            f"{self.__class__.__name__}"
            f"(cutoff={self.cutoff:.2f})"
        )

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            reference = load_cplx(
                self.reference_path,
                cleanup_first=True,
            )

            candidate = mmcif_to_complex(
                input.path_prefix + ".cif",
                selected_chains=(
                    input.tgt_chains + input.lig_chains
                ),
            )

            # 1. 利用目标蛋白 CA 完成整体对齐
            reference_target = target_ca_dict(
                reference,
                input.tgt_chains,
            )
            candidate_target = target_ca_dict(
                candidate,
                input.tgt_chains,
            )

            common_keys = [
                key
                for key in reference_target
                if key in candidate_target
            ]

            if len(common_keys) < 3:
                raise ValueError(
                    "fewer than three matching target CA atoms"
                )

            mobile_target = np.stack([
                candidate_target[key] for key in common_keys
            ])
            fixed_target = np.stack([
                reference_target[key] for key in common_keys
            ])

            rotation, translation = align_to_reference(
                mobile_target,
                fixed_target,
            )

            # 2. 提取参考和候选中的 B6、B7 主链原子
            reference_motif = motif_coords(
                reference,
                self.ligand_chain,
                self.anchor_positions,
                self.atom_names,
                self.expected_residues,
            )

            candidate_motif = motif_coords(
                candidate,
                self.ligand_chain,
                self.anchor_positions,
                self.atom_names,
                self.expected_residues,
            )

            candidate_motif_aligned = (
                rotation @ candidate_motif.T
            ).T + translation

            # 3. 计算两个 Gly 共8个主链原子的 RMSD
            diff = candidate_motif_aligned - reference_motif
            rmsd = float(
                np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
            )

            details = {
                "gg_backbone_rmsd": rmsd,
                "gg_rmsd_cutoff": self.cutoff,
            }

            if rmsd <= self.cutoff:
                return FilterResult.PASSED, details

            return FilterResult.FAILED, details

        except Exception as exc:
            return FilterResult.ERROR, {
                "gg_rmsd_error": str(exc),
            }
