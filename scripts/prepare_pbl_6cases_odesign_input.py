#!/usr/bin/env python3
"""Build ODesign input JSON for ligand-design 6-pocket cases.

Reference format follows ODesign/examples/ligand_design/prot_binding_lig/odesign_input.json.
For each *_pocket.pdb:
  - All protein residues in the reference pocket file are used as proteinChain sequence
    and full hotspot. Protein-like HETATM residues such as MSE are included.
  - Ligand length is taken from the official RCSB ligand definition for the bound
    ligand in that PDB entry, using heavy-atom count rather than total atom count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTEIN_LIKE_HET_RESNAMES = {"MSE", "SEC", "PYL"}

# Heavy-atom counts are derived from the official RCSB ligand formulas for each entry.
LIGAND_METADATA = {
    "2vt4": {
        "pdb_id": "2VT4",
        "ligand_code": "SOG",
        "ligand_name": "octyl 1-thio-beta-D-glucopyranoside",
        "heavy_atom_count": 20,
    },
    "5sdv": {
        "pdb_id": "5SDV",
        "ligand_code": "IAI",
        "ligand_name": (
            "1-methyl-N~4~-[(1,3-oxazol-4-yl)methyl]-N~5~-"
            "[(4R)-2-phenylimidazo[1,2-a]pyrimidin-7-yl]-1H-pyrazole-4,5-dicarboxamide"
        ),
        "heavy_atom_count": 33,
    },
    "6cm4": {
        "pdb_id": "6CM4",
        "ligand_code": "8NU",
        "ligand_name": (
            "3-[2-[4-(6-fluoranyl-1,2-benzoxazol-3-yl)piperidin-1-yl]ethyl]-"
            "2-methyl-6,7,8,9-tetrahydropyrido[1,2-a]pyrimidin-4-one"
        ),
        "heavy_atom_count": 30,
    },
    "7bkc": {
        "pdb_id": "7BKC",
        "ligand_code": "FAD",
        "ligand_name": "FLAVIN-ADENINE DINUCLEOTIDE",
        "heavy_atom_count": 53,
    },
    "7c7m": {
        "pdb_id": "7C7M",
        "ligand_code": "SAM",
        "ligand_name": "S-ADENOSYLMETHIONINE",
        "heavy_atom_count": 27,
    },
    "7v11": {
        "pdb_id": "7V11",
        "ligand_code": "OQO",
        "ligand_name": (
            "5-[1-[(1~{R})-1-[5-[3-chloranyl-2-fluoranyl-6-(1,2,3,4-tetrazol-1-yl)"
            "phenyl]-1-oxidanyl-pyridin-2-yl]-2-cyclopropyl-ethyl]pyrazol-4-yl]-"
            "6-methyl-pyridin-2-amine"
        ),
        "heavy_atom_count": 38,
    },
}


def _residue_ranges(chain_to_residues: dict[str, set[int]]) -> str:
    ranges: list[str] = []
    for chain_id in sorted(chain_to_residues):
        residue_ids = sorted(chain_to_residues[chain_id])
        start = prev = residue_ids[0]
        for residue_id in residue_ids[1:]:
            if residue_id == prev + 1:
                prev = residue_id
                continue
            ranges.append(f"{chain_id}/{start}-{prev}")
            start = prev = residue_id
        ranges.append(f"{chain_id}/{start}-{prev}")
    return ",".join(ranges)


def _hotspot_list(chain_to_residues: dict[str, set[int]]) -> str:
    return ",".join(
        f"{chain_id}/{residue_id}"
        for chain_id in sorted(chain_to_residues)
        for residue_id in sorted(chain_to_residues[chain_id])
    )


def _parse_single_pocket(pdb_path: Path) -> dict:
    chain_to_protein_residues: dict[str, set[int]] = {}

    with pdb_path.open() as handle:
        for line in handle:
            if len(line) < 54:
                continue
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue

            chain_id = line[21].strip() or "A"
            resn = line[17:20].strip()
            residue_raw = line[22:26].strip()
            try:
                residue_id = int(residue_raw)
            except ValueError:
                continue

            if record == "ATOM" or resn in PROTEIN_LIKE_HET_RESNAMES:
                chain_to_protein_residues.setdefault(chain_id, set()).add(residue_id)

    if not chain_to_protein_residues:
        raise ValueError(f"{pdb_path}: no protein ATOM residues found")

    case_name = pdb_path.stem.replace("_pocket", "")
    ligand_meta = LIGAND_METADATA.get(case_name)
    if ligand_meta is None:
        raise KeyError(f"{pdb_path}: missing ligand metadata for case '{case_name}'")

    sequence = _residue_ranges(chain_to_protein_residues)
    hotspot = _hotspot_list(chain_to_protein_residues)
    return {
        "name": case_name,
        "ref_file": f"./examples/protein_binding_ligand/6cases_pocket/6cases_pocket/{pdb_path.name}",
        "chains": [
            {"chain_type": "proteinChain", "sequence": sequence},
            {
                "chain_type": "ligand",
                "sequence": (
                    f"{ligand_meta['heavy_atom_count']}-{ligand_meta['heavy_atom_count']}"
                ),
            },
        ],
        "hotspot": hotspot,
        "center_method": "hotspot_center",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pocket-dir",
        type=Path,
        default=Path("examples/protein_binding_ligand/6cases_pocket/6cases_pocket"),
        help="Directory containing *_pocket.pdb files",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("examples/protein_binding_ligand/6cases_pocket/odesign_input.json"),
        help="Path to write generated ODesign input JSON",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=6,
        help="Expected number of *_pocket.pdb cases",
    )
    args = parser.parse_args()

    pocket_dir = args.pocket_dir.resolve()
    output_json = args.output_json.resolve()

    if not pocket_dir.is_dir():
        raise FileNotFoundError(f"pocket dir not found: {pocket_dir}")

    pocket_files = sorted(pocket_dir.glob("*_pocket.pdb"))
    if len(pocket_files) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} pocket files, got {len(pocket_files)} in {pocket_dir}"
        )

    records = []
    for path in pocket_files:
        record = _parse_single_pocket(path)
        records.append(record)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, indent=2))

    print(f"Wrote {output_json}")
    for record in records:
        ligand_meta = LIGAND_METADATA[record["name"]]
        print(
            f"  - {record['name']}: {ligand_meta['ligand_code']} "
            f"({ligand_meta['heavy_atom_count']} heavy atoms)"
        )


if __name__ == "__main__":
    main()
