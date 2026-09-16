"""Stage 5 composite scoring engine with configurable per-role hard floors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field
from m2a.scoring.clap_sim import ClapScorer, ClapSimilarityResult
from m2a.scoring.notes import NoteFidelityResult, evaluate_notes
from m2a.scoring.rhythm import RhythmScoreResult, evaluate_rhythm


class CompositeScoreResult(BaseModel):
    """Aggregate candidate assessment detailing component sub-metrics and gate statuses."""

    role: str
    composite_score: float = Field(ge=0.0, le=1.0)
    passed_hard_floors: bool
    rhythm_floor_passed: bool
    note_floor_passed: bool
    rejection_reason: str = ""
    rhythm: RhythmScoreResult
    notes: NoteFidelityResult
    clap: ClapSimilarityResult


DEFAULT_ROLE_CONFIGS = {
    "drums": {
        "weights": {"rhythm": 0.60, "notes": 0.10, "clap": 0.30},
        "floors": {"rhythm_f1": 0.85, "note_chroma": 0.40},
    },
    "bass": {
        "weights": {"rhythm": 0.40, "notes": 0.40, "clap": 0.20},
        "floors": {"rhythm_f1": 0.75, "note_chroma": 0.70},
    },
    "comp": {
        "weights": {"rhythm": 0.30, "notes": 0.45, "clap": 0.25},
        "floors": {"rhythm_f1": 0.70, "note_chroma": 0.70},
    },
    "lead": {
        "weights": {"rhythm": 0.20, "notes": 0.60, "clap": 0.20},
        "floors": {"rhythm_f1": 0.65, "note_chroma": 0.80},
    },
    "pads": {
        "weights": {"rhythm": 0.10, "notes": 0.40, "clap": 0.50},
        "floors": {"rhythm_f1": 0.40, "note_chroma": 0.60},
    },
}


class CandidateScorer:
    """Master multi-axis scoring coordinator enforcing rejection gates."""

    def __init__(self, clap_scorer: ClapScorer, config_overrides: Dict[str, Any] | None = None):
        self.clap_scorer = clap_scorer
        self.role_configs = DEFAULT_ROLE_CONFIGS.copy()
        if config_overrides:
            self.role_configs.update(config_overrides)

    def score_candidate(
        self,
        reference_audio_path: Path | str,
        candidate_audio_path: Path | str,
        role: str,
        prompt: str,
        negative_prompt: str = "",
    ) -> CompositeScoreResult:
        """Evaluate a generative candidate against the deterministic ground truth stem."""
        cfg = self.role_configs.get(role, self.role_configs["comp"])
        weights = cfg["weights"]
        floors = cfg["floors"]

        rhythm_res = evaluate_rhythm(reference_audio_path, candidate_audio_path, role=role)
        note_res = evaluate_notes(reference_audio_path, candidate_audio_path, floor_threshold=floors["note_chroma"])
        clap_res = self.clap_scorer.evaluate_candidate(candidate_audio_path, prompt, negative_prompt)

        # Evaluate Hard Floors
        rhythm_floor_passed = rhythm_res.f_measure >= floors["rhythm_f1"]
        note_floor_passed = note_res.chroma_cosine >= floors["note_chroma"]
        passed_hard_floors = rhythm_floor_passed and note_floor_passed

        rejection_reasons = []
        if not rhythm_floor_passed:
            rejection_reasons.append(
                f"Rhythm F1 ({rhythm_res.f_measure:.3f}) fell below hard floor ({floors['rhythm_f1']:.3f})"
            )
        if not note_floor_passed:
            rejection_reasons.append(
                f"Note Chroma ({note_res.chroma_cosine:.3f}) fell below hard floor ({floors['note_chroma']:.3f})"
            )

        # Normalize metrics to [0, 1] range
        normalized_clap = max(0.0, min(1.0, (clap_res.net_score + 1.0) / 2.0))
        normalized_chroma = max(0.0, min(1.0, (note_res.chroma_cosine + 1.0) / 2.0))

        raw_composite = (
            weights["rhythm"] * rhythm_res.f_measure
            + weights["notes"] * normalized_chroma
            + weights["clap"] * normalized_clap
        )

        final_composite = float(raw_composite) if passed_hard_floors else 0.0

        return CompositeScoreResult(
            role=role,
            composite_score=final_composite,
            passed_hard_floors=passed_hard_floors,
            rhythm_floor_passed=rhythm_floor_passed,
            note_floor_passed=note_floor_passed,
            rejection_reason="; ".join(rejection_reasons),
            rhythm=rhythm_res,
            notes=note_res,
            clap=clap_res,
        )