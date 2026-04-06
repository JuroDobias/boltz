from pathlib import Path

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    # Resolve output alignment template paths relative to the YAML file location.
    output = data.get("output") if isinstance(data, dict) else None
    if isinstance(output, dict):
        align = output.get("align_to_template")
        if isinstance(align, dict):
            for key in ("pdb", "cif"):
                raw = align.get(key)
                if raw:
                    p = Path(str(raw))
                    if not p.is_absolute():
                        align[key] = str((path.parent / p).resolve())

    template_ligand = data.get("template_ligand") if isinstance(data, dict) else None
    if isinstance(template_ligand, dict):
        for key in ("pdb", "cif", "sdf"):
            raw = template_ligand.get(key)
            if raw:
                p = Path(str(raw))
                if not p.is_absolute():
                    template_ligand[key] = str((path.parent / p).resolve())

    name = path.stem
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)
