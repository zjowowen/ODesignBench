import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.motif_scaffolding.motif_bench import MotifBenchEvaluator


class _AlwaysFailFoldseek:
    def foldseek_cluster(self, **kwargs):
        raise RuntimeError("foldseek unavailable")


def test_diversity_returns_zero_when_foldseek_fails_for_all_thresholds(tmp_path):
    evaluator = MotifBenchEvaluator.__new__(MotifBenchEvaluator)
    evaluator.logger = logging.getLogger("test_motif_bench")
    evaluator.du = _AlwaysFailFoldseek()

    result = evaluator._calculate_diversity_with_alpha5(
        successful_dir=Path(tmp_path),
        assist_protein=Path(tmp_path) / "assist.pdb",
        foldseek_db=str(tmp_path / "foldseek_db"),
    )

    assert result == {
        "Diversity": 0,
        "Clusters": 0,
        "Samples": 0,
        "Alpha5_Clusters": 0,
    }
