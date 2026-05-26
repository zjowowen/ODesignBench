from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_design.policy.hermes import HermesAgentPolicy
from agentic_design.problem_registry import load_all_problems
from agentic_design.proposal import make_strategy
from agentic_design.runtime.episode_runner import EpisodeRunner
from agentic_design.schemas import (
    ActionArtifacts,
    ActionMetrics,
    EpisodeConfig,
    EvaluationDefaults,
    ExternalPaths,
    LLMConfig,
    ODesignDefaults,
    PolicyConfig,
    ProposalStrategyConfig,
    ProtocolConfig,
)


def _write_problem_fixtures(root: Path) -> ExternalPaths:
    contig_csv = root / "contig.csv"
    contig_csv.write_text("problem,length,contig\n01_1LDB,125,0-100;A1-21;0-100\n", encoding="utf-8")
    baseline_json = root / "baseline.json"
    baseline_json.write_text(
        json.dumps(
            [
                {
                    "name": "01_1LDB",
                    "ref_file": str(root / "01_1LDB.pdb"),
                    "chains": [{"sequence": "52-52,A/1-21,52-52", "length": 125}],
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline_summary_csv = root / "baseline_summary.csv"
    baseline_summary_csv.write_text(
        "Problem,Success_Rate,Num_Solutions,Novelty\n01_1LDB,2.0,2,0.7\n",
        encoding="utf-8",
    )
    motif_dir = root / "motifs"
    motif_dir.mkdir()
    (motif_dir / "01_1LDB.pdb").write_text("HEADER TEST\n", encoding="utf-8")
    return ExternalPaths(
        odesign_pipeline_root=root,
        odesignbench_root=root,
        contig_csv=contig_csv,
        baseline_json=baseline_json,
        baseline_summary_csv=baseline_summary_csv,
        motif_pdbs_dir=motif_dir,
        esmfold_weights_dir=root / "esmfold",
        proteinmpnn_checkpoint=root / "proteinmpnn.pt",
    )


class FakeEngineManager:
    def __init__(self):
        self.ready_called = False
        self.shutdown_called = False
        self.odesign = object()
        self.proteinmpnn = object()
        self.esmfold = object()

    def ensure_ready(self):
        self.ready_called = True

    def shutdown(self):
        self.shutdown_called = True


class FakePolicy:
    def __init__(self):
        self._last = ("", "")

    def propose(self, episode_state, proposal_spec, motif_chains_lengths=None):
        self._last = (
            "system",
            f"user mode={proposal_spec.mode}",
        )
        if proposal_spec.mode == "screen-slate":
            return json.dumps(
                {
                    "mode": "screen-slate",
                    "actions": [
                        {"sequence": "0-0,A/1-21,104-104", "reasoning": "left extreme"},
                        {"sequence": "52-52,A/1-21,52-52", "reasoning": "center"},
                        {"sequence": "104-104,A/1-21,0-0", "reasoning": "right extreme"},
                    ],
                }
            )
        return json.dumps(
            {
                "mode": proposal_spec.mode,
                "actions": [
                    {"sequence": "52-52,A/1-21,52-52", "reasoning": "default"}
                ],
            }
        )

    def last_prompts(self):
        return self._last


class FakeRunODesignTool:
    def run_round(self, problem, slate, round_dir):
        artifacts = []
        for action in slate.actions:
            artifacts.append(
                ActionArtifacts(
                    action_id=action.action_id,
                    input_json=round_dir / "input.json",
                    infer_dir=round_dir / "infer" / action.action_id,
                    scaffold_dir=round_dir / "scaffolds" / action.action_id,
                    scaffold_count=action.n_sample,
                )
            )
        return artifacts


class FakeEvaluateDesignTool:
    def evaluate_round(self, problem, artifacts, round_dir):
        del problem
        del round_dir
        metrics = []
        for index, artifact in enumerate(artifacts):
            metrics.append(
                ActionMetrics(
                    action_id=artifact.action_id,
                    eval_dir=Path("/tmp") / artifact.action_id,
                    n_backbones=artifact.scaffold_count,
                    n_successes=1 if index == 0 else 0,
                    success_rate=100.0 if index == 0 else 0.0,
                    mean_rmsd=1.0 + index,
                    best_rmsd=1.0 + index,
                    mean_motif_rmsd=0.5 + index,
                    best_motif_rmsd=0.5 + index,
                )
            )
        return metrics


class AgenticDesignTests(unittest.TestCase):
    def test_screen_strategy_requires_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_problem_fixtures(Path(tmp))
            problem = load_all_problems(paths)["01_1LDB"]
            from agentic_design.schemas import BaselineSummary, EpisodeState

            state = EpisodeState(
                problem=problem,
                profile="explore",
                baseline=BaselineSummary(sequence=problem.baseline_sequence, success_rate=2.0, num_solutions=2, novelty=0.7),
            )
            strategy = make_strategy("screen-slate", {"slate_size": 3, "min_pairwise_linker_l1": 10})
            spec = strategy.build_spec(state, 16, problem)
            slate = strategy.parse_and_validate(
                json.dumps(
                    {
                        "mode": "screen-slate",
                        "actions": [
                            {"sequence": "0-0,A/1-21,104-104", "reasoning": "a"},
                            {"sequence": "52-52,A/1-21,52-52", "reasoning": "b"},
                            {"sequence": "104-104,A/1-21,0-0", "reasoning": "c"},
                        ],
                    }
                ),
                spec,
                state,
            )
            self.assertEqual(len(slate.actions), 3)
            self.assertEqual(slate.per_action_sample_budget, 5)

    def test_optimize_strategy_enforces_locality(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_problem_fixtures(Path(tmp))
            problem = load_all_problems(paths)["01_1LDB"]
            from agentic_design.schemas import BaselineSummary, EpisodeState, RoundRecord, ProposalSpec, ActionSlate, DesignAction

            incumbent = DesignAction(
                action_id="r00_a00",
                sequence="52-52,A/1-21,52-52",
                reasoning="incumbent",
                n_sample=16,
                n_step=200,
            )
            state = EpisodeState(
                problem=problem,
                profile="explore",
                baseline=BaselineSummary(sequence=problem.baseline_sequence, success_rate=2.0, num_solutions=2, novelty=0.7),
                rounds=[
                    RoundRecord(
                        round_index=0,
                        proposal_spec=ProposalSpec(
                            mode="single-action",
                            slate_size=1,
                            requested_sample_budget=16,
                            effective_sample_budget=16,
                            per_action_sample_budget=16,
                            action_schema={},
                            format_instructions="",
                            strategy_constraints={},
                            prompt_addendum="",
                        ),
                        action_slate=ActionSlate(
                            mode="single-action",
                            requested_sample_budget=16,
                            effective_sample_budget=16,
                            per_action_sample_budget=16,
                            actions=[incumbent],
                        ),
                        system_prompt="",
                        user_prompt="",
                        raw_output="{}",
                        artifacts=[],
                        metrics=[
                            ActionMetrics(
                                action_id="r00_a00",
                                eval_dir=Path("/tmp"),
                                n_backbones=16,
                                n_successes=1,
                                success_rate=6.25,
                                mean_rmsd=1.0,
                                best_rmsd=1.0,
                                mean_motif_rmsd=0.6,
                                best_motif_rmsd=0.6,
                            )
                        ],
                    )
                ],
            )
            strategy = make_strategy("optimize-slate", {"slate_size": 3, "max_pairwise_linker_l1": 6, "max_linker_slot_delta": 3})
            spec = strategy.build_spec(state, 16, problem)
            with self.assertRaises(ValueError):
                strategy.parse_and_validate(
                    json.dumps(
                        {
                            "mode": "optimize-slate",
                            "actions": [
                                {"sequence": "40-40,A/1-21,64-64", "reasoning": "too far"},
                                {"sequence": "52-52,A/1-21,52-52", "reasoning": "same"},
                                {"sequence": "53-53,A/1-21,51-51", "reasoning": "near"},
                            ],
                        }
                    ),
                    spec,
                    state,
                )

    def test_hermes_policy_uses_proposal_spec_in_prompt(self):
        llm_cfg = LLMConfig(api_base="http://example.com/v1", model="hermes")
        policy = HermesAgentPolicy(
            llm_cfg,
            completion_fn=lambda system_prompt, user_prompt: json.dumps(
                {
                    "system_len": len(system_prompt),
                    "user_has_mode": "screen-slate" in user_prompt,
                }
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_problem_fixtures(Path(tmp))
            problem = load_all_problems(paths)["01_1LDB"]
            from agentic_design.schemas import BaselineSummary, EpisodeState, ProposalSpec

            state = EpisodeState(
                problem=problem,
                profile="explore",
                baseline=BaselineSummary(sequence=problem.baseline_sequence, success_rate=2.0, num_solutions=2, novelty=0.7),
            )
            spec = ProposalSpec(
                mode="screen-slate",
                slate_size=3,
                requested_sample_budget=16,
                effective_sample_budget=15,
                per_action_sample_budget=5,
                action_schema={"type": "object"},
                format_instructions="JSON only",
                strategy_constraints={"min_pairwise_linker_l1": 10},
                prompt_addendum="be diverse",
            )
            raw = policy.propose(state, spec)
            self.assertIn('"user_has_mode": true', raw)
            _, user_prompt = policy.last_prompts()
            self.assertIn("screen-slate", user_prompt)

    def test_episode_runner_uses_schedule_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_problem_fixtures(root)
            config = EpisodeConfig(
                problem="01_1LDB",
                profile="explore",
                n_rounds=2,
                gpu_id=0,
                output_dir=root / "outputs",
                policy=PolicyConfig(type="hermes", llm=LLMConfig(api_base="http://example.com/v1", model="hermes")),
                proposal_strategy=ProposalStrategyConfig(
                    type="single-action",
                    params={},
                    schedule=[],
                ),
                odesign_defaults=ODesignDefaults(seeds=[1], n_step=200),
                evaluation_defaults=EvaluationDefaults(max_backbones=100),
                external_paths=paths,
                protocols={
                    "explore": ProtocolConfig(sample_budget=16),
                    "report": ProtocolConfig(sample_budget=100),
                },
            )
            runner = EpisodeRunner(
                config,
                engine_manager=FakeEngineManager(),
                policy=FakePolicy(),
                run_odesign_tool=FakeRunODesignTool(),
                evaluate_design_tool=FakeEvaluateDesignTool(),
            )
            summary = runner.run()
            self.assertEqual(summary.n_rounds, 2)
            self.assertTrue((config.output_dir / "01_1LDB" / "run_summary.json").exists())
            self.assertEqual(summary.best_action_id, "r00_a00")


if __name__ == "__main__":
    unittest.main()
