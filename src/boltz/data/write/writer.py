import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from boltz.data import const
from boltz.data.parse.template_reader import load_template_ca_by_chain
from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
        extra_mols_dir: Optional[str] = None,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings
        self.extra_mols_dir = Path(extra_mols_dir) if extra_mols_dir is not None else None
        self._template_cache: dict[str, dict[str, dict[int, np.ndarray]]] = {}

    def _load_template_ca_coords(self, template_path: str) -> dict[str, dict[int, np.ndarray]]:
        cached = self._template_cache.get(template_path)
        if cached is not None:
            return cached
        ca_by_chain = load_template_ca_by_chain(template_path)
        self._template_cache[template_path] = ca_by_chain
        return ca_by_chain

    def _kabsch(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src_centroid = src.mean(axis=0)
        dst_centroid = dst.mean(axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid
        cov = src_centered.T @ dst_centered
        u, _, vt = np.linalg.svd(cov)
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1, :] *= -1
            rot = vt.T @ u.T
        trans = dst_centroid - (src_centroid @ rot)
        return rot.astype(np.float32), trans.astype(np.float32)

    def _align_coords_to_template(
        self,
        coords: np.ndarray,
        structure: Structure,
        chain_info: list,
        align_opts,
    ) -> np.ndarray:
        template_ca = self._load_template_ca_coords(align_opts.path)
        chain_name_to_idx = {
            info.chain_name: idx for idx, info in enumerate(chain_info)
        }
        query_pts = []
        template_pts = []
        chain_map = align_opts.chain_map or {}

        for info in chain_info:
            chain_name = info.chain_name
            chain_idx = chain_name_to_idx[chain_name]
            chain = structure.chains[chain_idx]
            if int(chain["mol_type"]) != const.chain_type_ids["PROTEIN"]:
                continue
            template_chain = chain_map.get(chain_name, chain_name)
            if template_chain not in template_ca:
                continue
            chain_residues = structure.residues[
                chain["res_idx"] : chain["res_idx"] + chain["res_num"]
            ]
            for chain_pos, residue in enumerate(chain_residues, start=1):
                atom_start = residue["atom_idx"]
                atom_end = atom_start + residue["atom_num"]
                ca_idx = None
                for atom_idx in range(atom_start, atom_end):
                    if str(structure.atoms[atom_idx]["name"]).strip().upper() == "CA":
                        ca_idx = atom_idx
                        break
                if ca_idx is None:
                    continue
                res_num = int(residue["res_idx"]) + 1
                template_coord = template_ca[template_chain].get(res_num)
                if template_coord is None:
                    template_coord = template_ca[template_chain].get(chain_pos)
                if template_coord is None:
                    continue
                query_pts.append(coords[ca_idx])
                template_pts.append(template_coord)

        if len(query_pts) < 3:
            return coords

        query_arr = np.asarray(query_pts, dtype=np.float32)
        template_arr = np.asarray(template_pts, dtype=np.float32)
        rot, trans = self._kabsch(query_arr, template_arr)
        return (coords @ rot) + trans

    def _load_extra_mols(self, record_id: str) -> dict:
        if self.extra_mols_dir is None:
            return {}
        extra_mol_path = self.extra_mols_dir / f"{record_id}.pkl"
        if not extra_mol_path.exists():
            return {}
        with extra_mol_path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301

    def _select_export_ranks(self, export_cfg, n_models: int) -> set[int]:
        mode = export_cfg.export
        if mode == "all":
            return set(range(n_models))
        if mode == "top1":
            return {0}
        if mode == "topk":
            assert export_cfg.top_k is not None
            return set(range(min(int(export_cfg.top_k), n_models)))
        return set()

    def _write_aligned_ligand_sdf(
        self,
        struct_dir: Path,
        record,
        structure: Structure,
        chain_info: list,
        model_rank: int,
        export_cfg,
        extra_mols: dict,
    ) -> None:
        if not export_cfg.enabled or export_cfg.ligand_id is None:
            return

        lig_chain_idx = None
        for idx, info in enumerate(chain_info):
            if info.chain_name == export_cfg.ligand_id:
                lig_chain_idx = idx
                break
        if lig_chain_idx is None:
            return

        lig_chain = structure.chains[lig_chain_idx]
        atom_start = lig_chain["atom_idx"]
        atom_end = atom_start + lig_chain["atom_num"]
        residues = structure.residues[
            lig_chain["res_idx"] : lig_chain["res_idx"] + lig_chain["res_num"]
        ]
        if len(residues) == 0:
            return
        res_name = str(residues[0]["name"])
        ref_mol = extra_mols.get(res_name)
        if ref_mol is None:
            return

        predicted_coords_by_name = {}
        for atom_idx in range(atom_start, atom_end):
            atom_name = str(structure.atoms[atom_idx]["name"]).strip()
            predicted_coords_by_name[atom_name] = np.asarray(
                structure.atoms[atom_idx]["coords"], dtype=np.float64
            )

        base_mol = Chem.Mol(ref_mol)
        if base_mol.GetNumConformers() == 0:
            conformer = Chem.Conformer(base_mol.GetNumAtoms())
            base_mol.AddConformer(conformer, assignId=True)
        conf = base_mol.GetConformer()
        for atom in base_mol.GetAtoms():
            atom_name = atom.GetProp("name") if atom.HasProp("name") else None
            if atom_name is None or atom_name not in predicted_coords_by_name:
                continue
            x, y, z = predicted_coords_by_name[atom_name]
            conf.SetAtomPosition(atom.GetIdx(), (float(x), float(y), float(z)))

        export_mol = base_mol
        smiles = export_cfg.smiles
        if smiles:
            template = Chem.MolFromSmiles(smiles)
            if template is not None:
                try:
                    template_no_h = Chem.RemoveHs(template)
                    base_no_h = Chem.RemoveHs(base_mol)
                    assigned = AllChem.AssignBondOrdersFromTemplate(
                        template_no_h,
                        base_no_h,
                    )
                    if assigned.GetNumAtoms() == base_no_h.GetNumAtoms():
                        conf_base = base_no_h.GetConformer()
                        conf_assigned = Chem.Conformer(assigned.GetNumAtoms())
                        for atom_idx in range(assigned.GetNumAtoms()):
                            pos = conf_base.GetAtomPosition(atom_idx)
                            conf_assigned.SetAtomPosition(atom_idx, pos)
                        assigned.RemoveAllConformers()
                        assigned.AddConformer(conf_assigned, assignId=True)
                    export_mol = assigned
                except Exception:
                    export_mol = base_mol

        if export_cfg.add_hydrogens:
            export_mol = Chem.AddHs(export_mol, addCoords=True)

        filename = export_cfg.file_pattern.format(
            record=record.id,
            rank=model_rank,
            ligand=export_cfg.ligand_id,
        )
        out_path = struct_dir / filename
        writer = Chem.SDWriter(str(out_path))
        writer.write(export_mol)
        writer.close()

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            # Update chain info
            chain_info = []
            for chain in structure.chains:
                old_chain_idx = chain_map[chain["asym_id"]]
                old_chain_info = record.chains[old_chain_idx]
                new_chain_info = replace(
                    old_chain_info,
                    chain_id=int(chain["asym_id"]),
                    valid=True,
                )
                chain_info.append(new_chain_info)

            output_opts = getattr(record, "output_options", None)
            align_opts = (
                output_opts.align_to_template
                if output_opts is not None
                else None
            )
            sdf_opts = (
                output_opts.aligned_ligand_sdf
                if output_opts is not None
                else None
            )
            export_ranks = (
                self._select_export_ranks(sdf_opts, coord.shape[0])
                if (sdf_opts is not None and sdf_opts.enabled)
                else set()
            )
            extra_mols = (
                self._load_extra_mols(record.id)
                if (sdf_opts is not None and sdf_opts.enabled)
                else {}
            )

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()
                if align_opts is not None:
                    try:
                        coord_unpad = self._align_coords_to_template(
                            coord_unpad,
                            structure,
                            chain_info,
                            align_opts,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(  # noqa: T201
                            "[output] align_to_template failed for "
                            f"{record.id}: {exc}. Writing unaligned output."
                        )

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                model_rank = idx_to_rank[model_idx]
                outname = f"{record.id}_model_{model_rank}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and model_rank == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                if model_rank in export_ranks and sdf_opts is not None:
                    self._write_aligned_ligand_sdf(
                        struct_dir=struct_dir,
                        record=record,
                        structure=new_structure,
                        chain_info=chain_info,
                        model_rank=model_rank,
                        export_cfg=sdf_opts,
                        extra_mols=extra_mols,
                    )

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{model_rank}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{model_rank}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{model_rank}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{model_rank}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())
                
            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                s = prediction["s"].cpu().numpy()
                z = prediction["z"].cpu().numpy()

                path = (
                    struct_dir
                    / f"embeddings_{record.id}.npz"
                )
                np.savez_compressed(path, s=s, z=z)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
