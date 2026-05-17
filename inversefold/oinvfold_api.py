import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
import biotite.structure as struc
import biotite.structure.io as io
from biotite.structure.io import pdb, pdbx


def _read_structure_any(path: Path):
    if path.suffix.lower() == ".cif":
        cif_file = pdbx.CIFFile.read(str(path))
        try:
            return pdbx.get_structure(cif_file, model=1, extra_fields=["b_factor"])
        except Exception:
            return pdbx.get_structure(cif_file, model=1)
    pdb_file = pdb.PDBFile.read(str(path))
    try:
        return pdb.get_structure(pdb_file, model=1, extra_fields=["b_factor"])
    except Exception:
        return pdb.get_structure(pdb_file, model=1)


def _infer_design_modality(atom_array) -> str:
    polymer = atom_array[~atom_array.hetero]
    if len(polymer) == 0:
        return "ligand"
    res = set(np.unique(np.char.upper(polymer.res_name)))
    rna = {"A", "C", "G", "U"}
    dna = {"DA", "DC", "DG", "DT"}
    if res and res.issubset(rna):
        return "rna"
    if res and res.issubset(dna):
        return "dna"
    return "protein"


def _select_single_design_chain(atom_array) -> str:
    chains = []
    for chain_id in np.unique(atom_array.chain_id):
        chain = atom_array[atom_array.chain_id == chain_id]
        if len(chain) == 0:
            continue
        if hasattr(chain, "b_factor"):
            non_het = chain[~chain.hetero]
            if len(non_het) > 0 and np.all(non_het.b_factor == 0.0):
                chains.append(str(chain_id))
    # Some CIFs do not carry B_iso_or_equiv; in biotite this becomes all-NaN.
    # In that case, treat non-hetero nucleic/polymer chains as candidates.
    if len(chains) == 0:
        fallback = []
        for chain_id in np.unique(atom_array.chain_id):
            chain = atom_array[atom_array.chain_id == chain_id]
            if len(chain) == 0:
                continue
            non_het = chain[~chain.hetero]
            if len(non_het) > 0:
                fallback.append(str(chain_id))
        if len(fallback) == 1:
            return fallback[0]

    if len(chains) != 1:
        raise ValueError(
            f"OInvFold expects exactly one design chain marked by b_factor=0, got {chains}"
        )
    return chains[0]


def _select_design_chain_by_modality(atom_array, modality: str) -> str:
    """
    Fallback chain selector when B-factor design marks are unavailable.
    Select exactly one chain consistent with requested modality.
    """
    mode = str(modality).lower()
    candidates = []
    for chain_id in np.unique(atom_array.chain_id):
        chain = atom_array[atom_array.chain_id == chain_id]
        if len(chain) == 0:
            continue
        non_het = chain[~chain.hetero]
        if len(non_het) == 0:
            continue
        res_set = set(np.unique(np.char.upper(non_het.res_name)))
        if mode == "rna" and res_set and res_set.issubset({"A", "C", "G", "U"}):
            candidates.append(str(chain_id))
        elif mode == "dna" and res_set and res_set.issubset({"DA", "DC", "DG", "DT"}):
            candidates.append(str(chain_id))
        elif mode == "protein":
            # Treat non-RNA/DNA polymer as protein-like.
            if not (res_set and (res_set.issubset({"A", "C", "G", "U"}) or res_set.issubset({"DA", "DC", "DG", "DT"}))):
                candidates.append(str(chain_id))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot uniquely infer design chain for modality={mode}; candidates={candidates}"
    )


def _pick_primary_polymer_chain(atom_array) -> str:
    """
    Pick one non-hetero chain as receptor context for ligand atom-typing output.
    """
    best_chain = None
    best_len = -1
    for chain_id in np.unique(atom_array.chain_id):
        chain = atom_array[atom_array.chain_id == chain_id]
        if len(chain) == 0:
            continue
        polymer = chain[~chain.hetero]
        if len(polymer) == 0:
            continue
        if len(polymer) > best_len:
            best_len = len(polymer)
            best_chain = str(chain_id)
    if best_chain is None:
        raise ValueError("No polymer chain found for ligand-context output")
    return best_chain


def _ensure_cif_input(struct_path: Path, atom_array, tmp_dir: Path) -> Path:
    """
    OInvFold ligand CIF writers expect mmCIF input. Convert from PDB when needed.
    """
    if struct_path.suffix.lower() == ".cif":
        return struct_path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cif_path = tmp_dir / f"{struct_path.stem}.cif"
    io.save_structure(str(cif_path), atom_array)
    return cif_path


def _replace_chain_sequence(atom_array, chain_id: str, sequence: str, modality: str):
    seq = sequence.upper()
    out = atom_array.copy()
    starts = struc.get_residue_starts(out, add_exclusive_stop=True)
    idx = 0
    for s, e in zip(starts[:-1], starts[1:]):
        if str(out.chain_id[s]) != chain_id:
            continue
        if bool(out.hetero[s]):
            continue
        if idx >= len(seq):
            break
        aa = seq[idx]
        if modality == "protein":
            aa1_to_aa3 = {
                "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
                "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
                "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
                "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
            }
            out.res_name[s:e] = aa1_to_aa3.get(aa, "UNK")
        elif modality == "dna":
            base = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}.get(aa, "DA")
            out.res_name[s:e] = base
        elif modality == "rna":
            base = aa.replace("T", "U")
            out.res_name[s:e] = base if base in {"A", "C", "G", "U"} else "A"
        idx += 1
    return out


class OInvFoldRunner:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def _resolve_checkpoint(self, modality: str) -> str:
        ckpt_root = Path(str(getattr(self.cfg.inversefold, "oinvfold_ckpt_root", "ckpt")))
        # repo-local relative path support
        if not ckpt_root.is_absolute():
            ckpt_root = Path(__file__).resolve().parent.parent / ckpt_root
        ckpt = ckpt_root / f"oinvfold_{modality}.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"OInvFold checkpoint not found: {ckpt}. "
                "Please download oinvfold_{protein,ligand,dna,rna}.ckpt first."
            )
        return str(ckpt)

    @staticmethod
    def _default_model_cfg(modality: str) -> Dict[str, Any]:
        name = str(modality).lower()
        if name == "protein":
            dataset = "protein"
        elif name == "rna":
            dataset = "RNA"
        elif name == "dna":
            dataset = "DNA"
        elif name == "ligand":
            dataset = "ligand"
        else:
            dataset = "DNA"
        return {
            "dataset": dataset,
            "dataname": dataset,
            "model_name": "OInvFold",
            "geo_layer": 3,
            "attn_layer": 3,
            "node_layer": 3,
            "edge_layer": 3,
            "encoder_layer": 12,
            "hidden_dim": 128,
            "dropout": 0.0,
            "mask_rate": 0.1,
            "k_neighbors": 30,
            "virtual_frame_num": 8,
            "virtual_atom_num": 3,
            "steps_per_epoch": 7006,
        }

    def run(
        self,
        input_dir: Path,
        output_dir: str,
        gpu_list: List[str],
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        use_beam: Optional[bool] = None,
    ) -> Dict[str, Any]:
        input_dir = Path(input_dir)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # Ensure local imports resolve:
        # inversefold/OInvFold/{evaluation_tools,src}
        from .OInvFold.evaluation_tools.tools import (
            reload_model,
            parse_invfold,
            inference,
        )
        from .OInvFold.evaluation_tools.inference_utils import (
            save_pair_with_predicted_ligand_cif,
        )

        files = sorted(input_dir.glob("*.cif")) + sorted(input_dir.glob("*.pdb"))
        if not files:
            raise FileNotFoundError(f"No structures under {input_dir}")

        topk_cfg = getattr(self.cfg.inversefold, "oinvfold_topk", None)
        temp_cfg = getattr(self.cfg.inversefold, "oinvfold_temp", None)
        beam_cfg = getattr(self.cfg.inversefold, "oinvfold_use_beam", None)
        n_samples = int(topk_cfg if n_samples is None else n_samples)
        if n_samples <= 0:
            n_samples = 1
        if temperature is None:
            temperature = float(temp_cfg) if temp_cfg is not None else 1.0
        else:
            temperature = float(temperature)
        if temperature <= 0:
            temperature = 1.0
        use_beam = bool(beam_cfg if use_beam is None else use_beam)

        gpu_id = 0
        if gpu_list:
            try:
                gpu_id = int(str(gpu_list[0]))
            except Exception:
                gpu_id = 0
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        success = 0
        failed = 0
        outputs = 0
        failures: List[str] = []
        model_cache: Dict[str, Any] = {}
        tmp_cif_dir = output_root / "_oinvfold_tmp"

        for struct_path in files:
            try:
                atom_array = _read_structure_any(struct_path)
                modality = str(getattr(self.cfg.inversefold, "data_name", "auto")).lower()
                if modality == "auto":
                    modality = _infer_design_modality(atom_array)

                # OInvFold currently supports protein/rna/dna/ligand
                if modality not in {"protein", "rna", "dna", "ligand"}:
                    raise ValueError(f"Unsupported OInvFold modality: {modality}")

                if modality not in model_cache:
                    ckpt = self._resolve_checkpoint(modality)
                    ocfg = getattr(self.cfg.inversefold, "oinvfold_model_cfg", None)
                    if ocfg is None:
                        ocfg = OmegaConf.create(self._default_model_cfg(modality))
                    elif isinstance(ocfg, dict):
                        ocfg = OmegaConf.create(ocfg)
                    elif not hasattr(ocfg, "dataset"):
                        # Keep compatibility with plain mappings from Hydra containers.
                        ocfg = OmegaConf.create(dict(ocfg))
                    model, dev = reload_model(
                        modality,
                        checkpoint_path=ckpt,
                        configs=ocfg,
                        device=device,
                    )
                    model_cache[modality] = (model, dev)

                model, dev = model_cache[modality]
                pred = type("Pred", (), {"coordinate": torch.tensor(atom_array.coord[None, ...], dtype=torch.float32)})
                inv_samples = parse_invfold(
                    atom_array=atom_array,
                    pred_output=pred,
                    design_modality=modality,
                    sample_name=struct_path.stem,
                )
                if not inv_samples or not inv_samples[0]:
                    raise RuntimeError("parse_invfold produced no design samples")

                if modality == "ligand":
                    # Choose one ligand target sample (prefer the one with most atoms).
                    lig_samples = list(inv_samples[0])
                    lig_samples.sort(
                        key=lambda x: len(x.get("ligand", {}).get("elements", [])),
                        reverse=True,
                    )
                    smp = lig_samples[0]
                    pred_seqs, scores, _, _, _ = inference(
                        model=model,
                        sample_input=smp,
                        design_modality=modality,
                        topk=n_samples,
                        temp=temperature,
                        use_beam=use_beam,
                        device=dev,
                    )
                    if not pred_seqs:
                        raise RuntimeError("OInvFold returned empty ligand prediction")

                    rec_chain_id = _pick_primary_polymer_chain(atom_array)
                    lig_chain_id = str(smp.get("chain_id"))
                    lig_res_id = smp.get("res_id", None)
                    cif_src = _ensure_cif_input(struct_path, atom_array, tmp_cif_dir)

                    for i, seq in enumerate(pred_seqs[:n_samples], start=1):
                        pred_elements = [tok for tok in str(seq).split(" ") if tok]
                        out_path = output_root / f"{struct_path.stem}-{i}.cif"
                        save_pair_with_predicted_ligand_cif(
                            cif_src_path=str(cif_src),
                            rec_chain_id=rec_chain_id,
                            lig_chain_id=lig_chain_id,
                            lig_res_id=lig_res_id,
                            new_elems=pred_elements,
                            out_path=str(out_path),
                        )
                        outputs += 1
                else:
                    try:
                        design_chain = _select_single_design_chain(atom_array)
                    except Exception:
                        design_chain = _select_design_chain_by_modality(atom_array, modality)
                    # Choose sample matching the intended design chain.
                    chosen = None
                    for cand in inv_samples[0]:
                        title = str(cand.get("title", ""))
                        if f"chain{design_chain}" in title:
                            chosen = cand
                            break
                    if chosen is None:
                        chosen = inv_samples[0][0]
                    smp = chosen
                    pred_seqs, scores, _, _, _ = inference(
                        model=model,
                        sample_input=smp,
                        design_modality=modality,
                        topk=n_samples,
                        temp=temperature,
                        use_beam=use_beam,
                        device=dev,
                    )
                    if not pred_seqs:
                        raise RuntimeError("OInvFold returned empty prediction")

                    for i, seq in enumerate(pred_seqs[:n_samples], start=1):
                        out_arr = _replace_chain_sequence(atom_array, design_chain, seq, modality)
                        out_path = output_root / f"{struct_path.stem}-{i}.cif"
                        io.save_structure(str(out_path), out_arr)
                        outputs += 1
                success += 1
            except Exception as exc:
                failed += 1
                failures.append(f"{struct_path.name}: {exc}")
                print(f"[OInvFold] Failed on {struct_path.name}: {exc}")

        return {
            "success": success > 0 and outputs > 0,
            "stage": "inversefold.run_oinvfold",
            "outputs": {"output_dir": str(output_root)},
            "details": {
                "num_inputs": len(files),
                "num_success": success,
                "num_failed": failed,
                "num_outputs": outputs,
                "oinvfold_topk": n_samples,
                "oinvfold_temp": temperature,
                "oinvfold_use_beam": use_beam,
                "device": device,
                "failures": failures[:20],
            },
        }
