"""
Generate summary files for motif scaffolding evaluation.

Generates:
1. summary_by_problem.csv - Summary for each motif/problem
2. overall_summary.csv - Group-level statistics and MotifBench Score
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re
from typing import Optional


def parse_esm_summary(esm_summary_path: Path) -> dict:
    """Parse esm_summary.txt file and extract metrics."""
    if not esm_summary_path.exists():
        return None
    
    result = {}
    with open(esm_summary_path, 'r') as f:
        content = f.read()
        
        # Extract Evaluated Protein
        match = re.search(r'Evaluated Protein\s*\|\s*(\S+)', content)
        if match:
            result['Problem'] = match.group(1)
        
        # Extract Number of Unique Solutions
        match = re.search(r'Number of Unique Solutions[^(]*\([^)]*\)\s*\|\s*(\d+)', content)
        if match:
            result['Num_Solutions'] = int(match.group(1))
        else:
            # Try without parentheses
            match = re.search(r'Number of Unique Solutions\s*\|\s*(\d+)', content)
            if match:
                result['Num_Solutions'] = int(match.group(1))
        
        # Extract Novelty
        match = re.search(r'Novelty\s*\|\s*([\d.]+)', content)
        if match:
            result['Novelty'] = float(match.group(1))
        
        # Extract Success Rate
        match = re.search(r'Success Rate\s*\|\s*([\d.]+)', content)
        if match:
            result['Success_Rate'] = float(match.group(1))
        
        # Extract Number of Scaffolds Evaluated
        match = re.search(r'Number of Scaffolds Evaluated\s*\|\s*(\d+)', content)
        if match:
            result['Num_Evaluated'] = int(match.group(1))
    
    return result if result else None


def generate_summary_by_problem(base_output_dir: Path, output_path: Optional[Path] = None) -> Path:
    """
    Generate summary_by_problem.csv from all esm_summary.txt files.
    
    Args:
        base_output_dir: Base output directory containing motif subdirectories
        
    Returns:
        Path to generated summary_by_problem.csv
    """
    summary_by_problem_path = output_path if output_path is not None else (base_output_dir / "summary_by_problem.csv")
    
    # Find all esm_summary.txt files
    # Try motif_scaffolding/*/esm_summary.txt first (ODesignBench structure)
    motif_scaffolding_dir = base_output_dir / "motif_scaffolding"
    if motif_scaffolding_dir.exists():
        esm_summaries = list(motif_scaffolding_dir.glob("*/esm_summary.txt"))
    else:
        # Try direct structure: */esm_summary.txt (MotifBench structure)
        esm_summaries = list(base_output_dir.glob("*/esm_summary.txt"))
    
    results = []
    for esm_summary_path in sorted(esm_summaries):
        parsed = parse_esm_summary(esm_summary_path)
        if parsed:
            results.append(parsed)
    
    if not results:
        # Create empty file with headers
        df = pd.DataFrame(columns=['Problem', 'Num_Solutions', 'Novelty', 'Success_Rate', 'Num_Evaluated'])
        df.to_csv(summary_by_problem_path, index=False)
        return summary_by_problem_path
    
    df = pd.DataFrame(results)
    
    # Ensure all required columns exist
    required_cols = ['Problem', 'Num_Solutions', 'Novelty', 'Success_Rate', 'Num_Evaluated']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col != 'Novelty' else 0.0
    
    # Reorder columns
    df = df[required_cols]
    df.to_csv(summary_by_problem_path, index=False)
    
    return summary_by_problem_path


def generate_overall_summary(
    test_cases_path: Path,
    summary_by_problem_path: Path,
    overall_summary_path: Path
) -> Path:
    """
    Generate overall_summary.csv with group statistics and MotifBench Score.
    
    Args:
        test_cases_path: Path to test_cases.csv (contains Problem -> Group mapping)
        summary_by_problem_path: Path to summary_by_problem.csv
        overall_summary_path: Path to output overall_summary.csv
        
    Returns:
        Path to generated overall_summary.csv
    """
    if not test_cases_path.is_file():
        raise FileNotFoundError(
            f"Missing required test_cases.csv for motif group mapping: {test_cases_path}"
        )
    if test_cases_path.stat().st_size == 0:
        raise ValueError(
            f"test_cases.csv is empty and cannot be used for group mapping: {test_cases_path}"
        )

    test_cases = pd.read_csv(test_cases_path)
    required_columns = {"pdb_id", "group"}
    missing_columns = required_columns.difference(set(test_cases.columns))
    if missing_columns:
        raise ValueError(
            "test_cases.csv missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if test_cases.empty:
        raise ValueError(f"test_cases.csv has no rows: {test_cases_path}")

    # Load test cases to get group mapping
    problem_to_group = {}
    test_cases = test_cases.copy()
    test_cases['idx'] = [i + 1 for i in range(len(test_cases))]

    # Create mapping from Problem name (e.g., "01_1LDB") to group
    # Problem format: "01_1LDB" -> extract "01" -> idx=1 -> group
    for _, row in test_cases.iterrows():
        idx = row['idx']
        pdb_id = row.get('pdb_id', '')
        problem_name = f"{idx:02d}_{pdb_id}"
        problem_to_group[problem_name] = row.get('group', 'unknown')
        # Also add just the PDB ID as fallback
        problem_to_group[pdb_id] = row.get('group', 'unknown')
    
    # Load summary by problem
    summary_by_problem = pd.read_csv(summary_by_problem_path)
    
    # Add group column
    def get_group(problem_name):
        # Try exact match first
        if problem_name in problem_to_group:
            return problem_to_group[problem_name]
        # Try extracting PDB ID (e.g., "01_1LDB" -> "1LDB")
        if '_' in str(problem_name):
            pdb_id = str(problem_name).split('_', 1)[1]
            if pdb_id in problem_to_group:
                return problem_to_group[pdb_id]
        return 'unknown'
    
    summary_by_problem['group'] = summary_by_problem['Problem'].apply(get_group)
    
    unmapped = summary_by_problem[summary_by_problem['group'] == 'unknown']['Problem'].tolist()
    if unmapped:
        raise ValueError(
            "Unmapped motif problems detected in summary generation. "
            "Please ensure test_cases.csv matches benchmark problems. "
            f"Unmapped: {', '.join(unmapped)}"
        )

    # Group by group and calculate statistics
    summary_by_group = summary_by_problem.groupby('group').agg(
        Number_Solved=('Num_Solutions', lambda x: (x > 0).sum()),
        Mean_Num_Solutions=('Num_Solutions', 'mean'),
        Mean_Novelty=('Novelty', 'mean'),
        Mean_Success_rate=('Success_Rate', 'mean')
    ).reset_index()

    summary_by_group.rename(columns={'group': 'Group'}, inplace=True)

    # Add overall row
    summary_by_group.loc[len(summary_by_group)] = {
        "Group": "overall",
        "Number_Solved": int(summary_by_group["Number_Solved"].sum()),
        "Mean_Num_Solutions": float(summary_by_group["Mean_Num_Solutions"].mean()),
        "Mean_Novelty": float(summary_by_group["Mean_Novelty"].mean()),
        "Mean_Success_rate": float(summary_by_group["Mean_Success_rate"].mean()),
    }
    
    # Calculate MotifBench Score
    # Formula: (100 + alpha) * Num_Solutions / (Num_Solutions + alpha) where alpha=5
    alpha = 5
    summary_by_problem['overall_score'] = (100 + alpha) * summary_by_problem['Num_Solutions'] / (
        summary_by_problem['Num_Solutions'] + alpha
    )
    motifbench_score = summary_by_problem['overall_score'].mean()
    
    # Write to file
    with open(overall_summary_path, 'w') as f:
        summary_by_group.to_csv(f, float_format='%.2f', index=False)
        f.write(f"\nMotifBench score: {motifbench_score:.2f}\n")
    
    return overall_summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary files for motif scaffolding evaluation"
    )
    parser.add_argument(
        "base_output_dir",
        help="Base output directory containing motif evaluation results"
    )
    parser.add_argument(
        "--test-cases",
        help="Path to test_cases.csv (optional, for group mapping)",
        default=None
    )
    parser.add_argument(
        "--summary-by-problem",
        help="Output path for summary_by_problem.csv (default: base_output_dir/summary_by_problem.csv)",
        default=None
    )
    parser.add_argument(
        "--overall-summary",
        help="Output path for overall_summary.csv (default: base_output_dir/overall_summary.csv)",
        default=None
    )
    
    args = parser.parse_args()
    
    base_output_dir = Path(args.base_output_dir)
    
    # Generate summary_by_problem.csv
    summary_by_problem_path = Path(args.summary_by_problem) if args.summary_by_problem else base_output_dir / "summary_by_problem.csv"
    generate_summary_by_problem(base_output_dir, output_path=summary_by_problem_path)
    
    # Generate overall_summary.csv
    test_cases_path = Path(args.test_cases) if args.test_cases else None
    overall_summary_path = Path(args.overall_summary) if args.overall_summary else base_output_dir / "overall_summary.csv"
    
    if not test_cases_path:
        raise ValueError(
            "--test-cases is required to avoid unknown motif groups in overall_summary.csv"
        )
    generate_overall_summary(test_cases_path, summary_by_problem_path, overall_summary_path)
    
    print(f"Summary files generated:")
    print(f"  - {summary_by_problem_path}")
    print(f"  - {overall_summary_path}")


if __name__ == "__main__":
    main()
