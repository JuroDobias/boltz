from pathlib import Path

import gemmi
import numpy as np


def _parse_res_idx(value: str) -> int | None:
    value = str(value).strip()
    if value in {"", ".", "?"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _atom_lookup_from_structure(path: str) -> dict[tuple[str, int, str], np.ndarray]:
    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    structure = gemmi.read_structure(path)
    if len(structure) == 0:
        return lookup
    model = next(iter(structure))
    for chain in model:
        chain_name = str(chain.name).strip()
        if chain_name in {"", ".", "?"}:
            continue
        for residue in chain:
            if len(residue) == 0:
                continue
            try:
                res_idx = int(residue.seqid.num)
            except Exception:
                continue
            for atom in residue:
                atom_name = str(atom.name).strip().upper()
                if atom_name in {"", ".", "?"}:
                    continue
                lookup[(chain_name, res_idx, atom_name)] = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z],
                    dtype=np.float32,
                )
    return lookup


def _tag_index(tags, name: str) -> int | None:
    for i, tag in enumerate(tags):
        if tag == name:
            return i
    return None


def _atom_lookup_from_cif_atom_site(path: str) -> dict[tuple[str, int, str], np.ndarray]:
    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    doc = gemmi.cif.read_file(path)
    block = doc.sole_block() if len(doc) == 1 else doc[0]
    atom_site = block.find_mmcif_category("_atom_site.")
    if not atom_site:
        return lookup

    i_atom = _tag_index(atom_site.tags, "_atom_site.label_atom_id")
    if i_atom is None:
        i_atom = _tag_index(atom_site.tags, "_atom_site.auth_atom_id")

    i_chain = _tag_index(atom_site.tags, "_atom_site.auth_asym_id")
    if i_chain is None:
        i_chain = _tag_index(atom_site.tags, "_atom_site.label_asym_id")

    i_seq = _tag_index(atom_site.tags, "_atom_site.label_seq_id")
    if i_seq is None:
        i_seq = _tag_index(atom_site.tags, "_atom_site.auth_seq_id")

    i_x = _tag_index(atom_site.tags, "_atom_site.Cartn_x")
    i_y = _tag_index(atom_site.tags, "_atom_site.Cartn_y")
    i_z = _tag_index(atom_site.tags, "_atom_site.Cartn_z")

    i_model = _tag_index(atom_site.tags, "_atom_site.pdbx_PDB_model_num")
    first_model = None
    required = (i_atom, i_chain, i_seq, i_x, i_y, i_z)
    if any(i is None for i in required):
        return lookup

    for row in atom_site:
        if i_model is not None:
            model_value = str(row[i_model]).strip()
            if model_value not in {"", ".", "?"}:
                if first_model is None:
                    first_model = model_value
                elif model_value != first_model:
                    continue

        atom_name = str(row[i_atom]).strip().upper()
        chain_name = str(row[i_chain]).strip()
        res_idx = _parse_res_idx(row[i_seq])
        if (
            atom_name in {"", ".", "?"}
            or chain_name in {"", ".", "?"}
            or res_idx is None
        ):
            continue
        try:
            x = float(row[i_x])
            y = float(row[i_y])
            z = float(row[i_z])
        except Exception:
            continue
        lookup[(chain_name, res_idx, atom_name)] = np.array([x, y, z], dtype=np.float32)
    return lookup


def load_template_atom_lookup(
    path: str,
    template_chain_id: str | None = None,
) -> dict[tuple[str, int, str], np.ndarray]:
    """Load template atom coordinates indexed by (chain, residue index, atom name)."""
    template_path = str(Path(path))
    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    try:
        lookup = _atom_lookup_from_structure(template_path)
    except Exception:
        lookup = {}

    if not lookup and template_path.lower().endswith((".cif", ".mmcif")):
        try:
            lookup = _atom_lookup_from_cif_atom_site(template_path)
        except Exception:
            lookup = {}

    if template_chain_id is not None:
        lookup = {
            key: value
            for key, value in lookup.items()
            if key[0] == template_chain_id
        }

    if not lookup:
        msg = f"Could not extract template atoms from '{template_path}'"
        raise ValueError(msg)
    return lookup


def load_template_ca_by_chain(path: str) -> dict[str, dict[int, np.ndarray]]:
    """Load template CA coordinates grouped by chain and residue index."""
    atom_lookup = load_template_atom_lookup(path)
    ca_by_chain: dict[str, dict[int, np.ndarray]] = {}
    for (chain_name, res_idx, atom_name), coord in atom_lookup.items():
        if atom_name == "CA":
            ca_by_chain.setdefault(chain_name, {})[res_idx] = coord

    if not ca_by_chain:
        msg = f"Could not extract CA atoms from template '{path}'"
        raise ValueError(msg)
    return ca_by_chain
