"""Stage 5: Successive-Halving Bandit search optimizer with financial circuit breaker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from m2a.prompts.generate import PromptVariant, generate_prompt_variants
from m2a.restyle.base import CandidateResult, EndpointErrorType, EndpointException, RestyleEndpoint
from m2a.scoring.combine import CandidateScorer, CompositeScoreResult


class BudgetTracker(BaseModel):
    """Financial circuit breaker monitoring total endpoint invocations and USD spend."""

    max_calls: int
    max_usd: float
    current_calls: int = 0
    current_usd: float = 0.0

    def can_spend(self, estimated_cost_usd: float) -> bool:
        """Check if executing another endpoint call violates budget constraints."""
        if self.current_calls >= self.max_calls:
            return False
        if (self.current_usd + estimated_cost_usd) > self.max_usd:
            return False
        return True

    def record_spend(self, actual_cost_usd: float) -> None:
        """Increment internal invocation count and spent financial balance."""
        self.current_calls += 1
        self.current_usd += actual_cost_usd


class EvaluatedCandidate(BaseModel):
    """Container associating a candidate generation with its composite evaluation score."""

    candidate: CandidateResult
    evaluation: CompositeScoreResult


class SearchSessionResult(BaseModel):
    """Complete summary of the successive-halving optimization run."""

    role: str
    best_candidate: Optional[EvaluatedCandidate] = None
    all_evaluated: List[EvaluatedCandidate] = Field(default_factory=list)
    total_calls: int = 0
    total_usd: float = 0.0
    budget_exhausted: bool = False


class SuccessiveHalvingOptimizer:
    """Explores (strength x prompt x seed) space with successive halving and hard-floor gating."""

    def __init__(
        self,
        endpoint: RestyleEndpoint,
        scorer: CandidateScorer,
        budget: BudgetTracker,
        coarse_strengths: Optional[List[float]] = None,
        coarse_seeds: Optional[List[int]] = None,
    ):
        self.endpoint = endpoint
        self.scorer = scorer
        self.budget = budget
        self.strengths = coarse_strengths or [0.60, 0.70, 0.80]
        self.seeds = coarse_seeds or [42, 101]

    def optimize_stem(
        self,
        stem_audio_path: Path | str,
        role: str,
        house_style_desc: str = "",
    ) -> SearchSessionResult:
        """Run two-round successive halving bandit search over parameter grid."""
        prompts = generate_prompt_variants(role, house_style_desc)
        all_evaluations: List[EvaluatedCandidate] = []
        budget_halted = False

        # -------------------------------------------------------------
        # ROUND 1: Coarse Grid Exploration
        # -------------------------------------------------------------
        round1_grid: List[Tuple[float, PromptVariant, int]] = []
        for s in self.strengths:
            for p in prompts:
                for seed in self.seeds[:1]:  # Use single seed for coarse filtering
                    round1_grid.append((s, p, seed))

        for strength, prompt_var, seed in round1_grid:
            est_cost = 0.05  # Approximate cost estimate
            if not self.budget.can_spend(est_cost):
                budget_halted = True
                break

            try:
                cand = self.endpoint.transform(
                    audio_path=stem_audio_path,
                    prompt=prompt_var.prompt,
                    negative_prompt=prompt_var.negative_prompt,
                    strength=strength,
                    seed=seed,
                )
                self.budget.record_spend(cand.cost_usd)
            except EndpointException as e:
                if e.error_type == EndpointErrorType.FATAL:
                    continue
                raise

            score_res = self.scorer.score_candidate(
                reference_audio_path=stem_audio_path,
                candidate_audio_path=cand.audio_path,
                role=role,
                prompt=prompt_var.prompt,
                negative_prompt=prompt_var.negative_prompt,
            )
            all_evaluations.append(EvaluatedCandidate(candidate=cand, evaluation=score_res))

        if not all_evaluations:
            return SearchSessionResult(
                role=role,
                best_candidate=None,
                all_evaluated=[],
                total_calls=self.budget.current_calls,
                total_usd=self.budget.current_usd,
                budget_exhausted=budget_halted,
            )

        # -------------------------------------------------------------
        # ROUND 2: Refine Top Quartile (Successive Halving)
        # -------------------------------------------------------------
        # Filter for candidates passing hard floors, then sort by composite score
        valid_candidates = [e for e in all_evaluations if e.evaluation.passed_hard_floors]
        pool_to_halve = valid_candidates if valid_candidates else all_evaluations
        pool_to_halve.sort(key=lambda x: x.evaluation.composite_score, reverse=True)

        k_survivors = max(1, len(pool_to_halve) // 2)
        survivors = pool_to_halve[:k_survivors]

        for surv in survivors:
            if budget_halted:
                break
            # Resample around survivor parameters with secondary seed
            resample_seed = surv.candidate.seed + 999
            if not self.budget.can_spend(0.05):
                budget_halted = True
                break

            try:
                cand2 = self.endpoint.transform(
                    audio_path=stem_audio_path,
                    prompt=surv.candidate.prompt,
                    negative_prompt=surv.candidate.negative_prompt,
                    strength=surv.candidate.strength,
                    seed=resample_seed,
                )
                self.budget.record_spend(cand2.cost_usd)

                score2 = self.scorer.score_candidate(
                    reference_audio_path=stem_audio_path,
                    candidate_audio_path=cand2.audio_path,
                    role=role,
                    prompt=surv.candidate.prompt,
                    negative_prompt=surv.candidate.negative_prompt,
                )
                all_evaluations.append(EvaluatedCandidate(candidate=cand2, evaluation=score2))
            except EndpointException:
                continue

        # Final ranking: pick highest-scoring candidate that passed hard floors
        passing = [e for e in all_evaluations if e.evaluation.passed_hard_floors]
        best = max(passing, key=lambda x: x.evaluation.composite_score) if passing else None

        return SearchSessionResult(
            role=role,
            best_candidate=best,
            all_evaluated=all_evaluations,
            total_calls=self.budget.current_calls,
            total_usd=self.budget.current_usd,
            budget_exhausted=budget_halted,
        )