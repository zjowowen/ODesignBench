#!/usr/bin/env python3
"""Generate the AF3 dialect patch by reading the source inside a Docker container.

This script is called by run_af3.sh. It runs inside a throwaway AF3 Docker container,
reads the original folding_input.py, applies the MSA-injection patch, and writes
the result to /tmp/alphafold3_folding_input_patched.py on the host (via -v /tmp:/tmp).
"""
import pathlib

src = pathlib.Path("/app/alphafold/src/alphafold3/common/folding_input.py")
dst = pathlib.Path("/tmp/alphafold3_folding_input_patched.py")
src_text = src.read_text()

# --- Patch 1: accept MSA fields in proteinChain.from_alphafoldserver_dict ---
old_validate = """    _validate_keys(
        json_dict.keys(),
        {
            'sequence',
            'glycans',
            'modifications',
            'count',
            'maxTemplateDate',
            'useStructureTemplate',
        },
    )"""
new_validate = """    _validate_keys(
        json_dict.keys(),
        {
            'sequence',
            'glycans',
            'modifications',
            'count',
            'maxTemplateDate',
            'useStructureTemplate',
            # MSA fields (accepted by patch, processed below)
            'unpairedMsa',
            'unpairedMsaPath',
            'pairedMsa',
            'pairedMsaPath',
            'templates',
        },
    )"""
assert old_validate in src_text, "proteinChain validate patch target not found"
src_text = src_text.replace(old_validate, new_validate, 1)

# --- Patch 2: read MSA from the accepted fields, set templates=[] ---
# When MSA fields are provided, templates must be set to [] (not None) because
# featurisation.py validates that templates is not None. We also avoid using
# the data pipeline templates.
old_after_templates = """    templates = None  # Search for templates unless explicitly disabled.
    if not json_dict.get('useStructureTemplate', True):
      templates = []  # Do not use any templates."""

new_after_templates = """    templates = None  # Search for templates unless explicitly disabled.
    if not json_dict.get('useStructureTemplate', True):
      templates = []  # Do not use any templates.

    # --- MSA patch: read pre-computed MSA from JSON fields ---
    unpaired_msa = json_dict.get('unpairedMsa', None)
    unpaired_msa_path = json_dict.get('unpairedMsaPath', None)
    if unpaired_msa and unpaired_msa_path:
      raise ValueError('Only one of unpairedMsa/unpairedMsaPath can be set.')
    if unpaired_msa_path:
      unpaired_msa_path_obj = pathlib.Path(unpaired_msa_path)
      if not unpaired_msa_path_obj.is_absolute():
        raise ValueError(
            'unpairedMsaPath must be an absolute path in the alphafoldserver dialect.'
        )
      with open(unpaired_msa_path_obj) as _f:
        unpaired_msa = _f.read()

    paired_msa = json_dict.get('pairedMsa', None)
    paired_msa_path = json_dict.get('pairedMsaPath', None)
    if paired_msa and paired_msa_path:
      raise ValueError('Only one of pairedMsa/pairedMsaPath can be set.')
    if paired_msa_path:
      paired_msa_path_obj = pathlib.Path(paired_msa_path)
      if not paired_msa_path_obj.is_absolute():
        raise ValueError(
            'pairedMsaPath must be an absolute path in the alphafoldserver dialect.'
        )
      with open(paired_msa_path_obj) as _f:
        paired_msa = _f.read()
    # When MSA is provided, disable template search (templates must not be None
    # for featurisation, so set to empty list).
    if unpaired_msa is not None or paired_msa is not None:
      templates = []  # --- end MSA patch ---"""

assert old_after_templates in src_text, "MSA injection patch location not found"
src_text = src_text.replace(old_after_templates, new_after_templates, 1)

# --- Patch 3: pass MSA to constructor ---
old_constructor = """    return cls(id=seq_id, sequence=sequence, ptms=ptms, templates=templates)"""
new_constructor = """    return cls(id=seq_id, sequence=sequence, ptms=ptms, templates=templates,
               unpaired_msa=unpaired_msa, paired_msa=paired_msa)"""

assert old_constructor in src_text, "constructor patch target not found"
src_text = src_text.replace(old_constructor, new_constructor, 1)

dst.write_text(src_text)
print(f"Patched {src} -> {dst}")
