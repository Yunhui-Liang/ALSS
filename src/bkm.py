"""Bisecting K-Means with cluster-selection strategies and ALSS lookahead."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

# Small candidate clusters are more efficient with one BLAS/OpenMP thread and
# avoid scikit-learn's documented MKL warning on Windows. Existing user values
# are respected.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from .evaluation import cluster_sse, sse_for_indices
from .split_rules import SplitProposal, propose_split, savaresi_shape_index, split_gain


class BKM:
    """Bisecting K-Means that repeatedly splits one active cluster.

    Parameters
    ----------
    n_clusters:
        Desired number of non-empty output clusters.
    strategy:
        One of ``largest``, ``inertia``, ``mean_sse``, ``random``, ``gain``,
        ``alss``, ``alss_relative``, or ``alss_sum``.
    lambda_:
        Weight of ALSS's one-step lookahead gain.
    horizon_aware:
        Disable lookahead when the current split consumes the final requested
        cluster split, because no future split remains in the BKM budget.
    top_r:
        Number of clusters retained by ALSS after screening by immediate gain.
    n_init:
        Number of k-means++ initializations used for each binary split.
    random_state:
        Base seed. Seeds for individual splits are derived deterministically.
    """

    STRATEGIES = {
        "largest", "inertia", "mean_sse", "random", "gain", "savaresi",
        "alss", "alss_relative", "alss_sum", "gain_relative",
        "alss_relative_screen", "alss_relative_score",
    }

    def __init__(
        self,
        n_clusters: int,
        strategy: str = "gain",
        lambda_: float = 1.0,
        top_r: int = 3,
        n_init: int = 10,
        random_state: int = 0,
        horizon_aware: bool = False,
    ) -> None:
        strategy = strategy.lower()
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(self.STRATEGIES)}")
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if top_r < 1:
            raise ValueError("top_r must be positive")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if n_init < 1:
            raise ValueError("n_init must be positive")

        self.n_clusters = int(n_clusters)
        self.strategy = strategy
        self.lambda_ = float(lambda_)
        self.top_r = int(top_r)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.horizon_aware = bool(horizon_aware)
        self.kmeans_fit_calls_ = 0

    def _seed(self, step: int, cluster_id: int, purpose: int = 0) -> int:
        return int(
            (self.random_state + 1_000_003 * step + 10_007 * cluster_id + 97 * purpose)
            % (2**32 - 1)
        )

    def _new_proposal(
        self, X: np.ndarray, cluster_id: int, members: np.ndarray, step: int
    ) -> SplitProposal:
        self.kmeans_fit_calls_ += 1
        return propose_split(
            X,
            members,
            cluster_id=cluster_id,
            n_init=self.n_init,
            random_state=self._seed(step, cluster_id),
        )

    def fit(self, X: np.ndarray) -> "BKM":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples")

        started = perf_counter()
        clusters: dict[int, np.ndarray] = {0: np.arange(X.shape[0], dtype=np.int64)}
        next_cluster_id = 1
        current_sse = cluster_sse(X, np.zeros(X.shape[0], dtype=np.int64))
        self.sse_history_ = [current_sse]
        self.trajectory_: list[dict[str, Any]] = []
        self.candidate_diagnostics_: list[dict[str, Any]] = []
        self.n_iter_ = 0
        self.kmeans_fit_calls_ = 0

        for step in range(self.n_clusters - 1):
            eligible = {
                cid: members for cid, members in clusters.items() if members.size >= 2
            }
            if not eligible:
                raise RuntimeError(
                    "Cannot reach the requested cluster count: every active cluster is a singleton"
                )

            proposal: SplitProposal
            future_gain: float | None = None
            score: float | None
            shortlist_size: int | None = None
            candidate_scores: dict[int, float | None] = {}
            lookahead_gains: dict[int, float | None] = {}
            lookahead_components: dict[int, tuple[float, float]] = {}
            lookahead_enabled = False

            if self.strategy in {"largest", "inertia", "mean_sse", "random"}:
                if self.strategy == "largest":
                    selected_id = min(eligible, key=lambda cid: (-eligible[cid].size, cid))
                    score = float(eligible[selected_id].size)
                    candidate_scores = {
                        cid: float(members.size) for cid, members in eligible.items()
                    }
                elif self.strategy == "inertia":
                    inertias = {
                        cid: sse_for_indices(X, members) for cid, members in eligible.items()
                    }
                    selected_id = min(inertias, key=lambda cid: (-inertias[cid], cid))
                    score = float(inertias[selected_id])
                    candidate_scores = inertias.copy()
                elif self.strategy == "mean_sse":
                    mean_sse = {
                        cid: sse_for_indices(X, members) / members.size
                        for cid, members in eligible.items()
                    }
                    selected_id = min(mean_sse, key=lambda cid: (-mean_sse[cid], cid))
                    score = float(mean_sse[selected_id])
                    candidate_scores = mean_sse.copy()
                else:
                    eligible_ids = sorted(eligible)
                    selected_id = int(
                        np.random.default_rng(self._seed(step, 0, purpose=2)).choice(
                            eligible_ids
                        )
                    )
                    score = None
                    candidate_scores = {cid: None for cid in eligible}
                proposal = self._new_proposal(
                    X, selected_id, eligible[selected_id], step
                )
            elif self.strategy == "gain":
                proposals = {
                    cid: self._new_proposal(X, cid, members, step)
                    for cid, members in eligible.items()
                }
                selected_id = min(
                    proposals,
                    key=lambda cid: (-proposals[cid].gain, cid),
                )
                proposal = proposals[selected_id]
                score = proposal.gain
                candidate_scores = {
                    cid: candidate.gain for cid, candidate in proposals.items()
                }
            elif self.strategy == "gain_relative":
                proposals = {
                    cid: self._new_proposal(X, cid, members, step)
                    for cid, members in eligible.items()
                }
                candidate_scores = {
                    cid: (
                        candidate.gain / candidate.parent_sse
                        if candidate.parent_sse > np.finfo(float).eps
                        else 0.0
                    )
                    for cid, candidate in proposals.items()
                }
                selected_id = min(
                    candidate_scores,
                    key=lambda cid: (-candidate_scores[cid], cid),
                )
                proposal = proposals[selected_id]
                score = candidate_scores[selected_id]
            elif self.strategy == "savaresi":
                screened_ids = sorted(
                    eligible,
                    key=lambda cid: (-eligible[cid].size, cid),
                )[:10]
                shortlist_size = len(screened_ids)
                proposals = {
                    cid: self._new_proposal(X, cid, eligible[cid], step)
                    for cid in screened_ids
                }
                raw_scores = {
                    cid: savaresi_shape_index(X, candidate)
                    for cid, candidate in proposals.items()
                }
                finite_scores = [value for value in raw_scores.values() if value is not None]
                invalid_score = (
                    max(finite_scores) + max(1.0, abs(max(finite_scores)) * 0.05)
                    if finite_scores
                    else 1.0
                )
                candidate_scores = {
                    cid: invalid_score if value is None else value
                    for cid, value in raw_scores.items()
                }
                selected_id = min(
                    candidate_scores,
                    key=lambda cid: (candidate_scores[cid], cid),
                )
                proposal = proposals[selected_id]
                score = candidate_scores[selected_id]
            else:
                proposals = {
                    cid: self._new_proposal(X, cid, members, step)
                    for cid, members in eligible.items()
                }
                if self.strategy in {"alss_relative", "alss_relative_screen"}:
                    screening_scores = {
                        cid: (
                            candidate.gain / candidate.parent_sse
                            if candidate.parent_sse > np.finfo(float).eps
                            else 0.0
                        )
                        for cid, candidate in proposals.items()
                    }
                else:
                    screening_scores = {
                        cid: candidate.gain for cid, candidate in proposals.items()
                    }
                screened_ids = sorted(
                    proposals,
                    key=lambda cid: (-screening_scores[cid], cid),
                )[: self.top_r]
                shortlist_size = len(screened_ids)
                scored: list[tuple[float, int, float]] = []
                lookahead_enabled = self.lambda_ > 0 and (
                    not self.horizon_aware or step < self.n_clusters - 2
                )
                for cid in screened_ids:
                    candidate = proposals[cid]
                    child_gains = (0.0, 0.0)
                    if not lookahead_enabled:
                        candidate_future = 0.0
                    else:
                        child_gains = (
                            split_gain(
                                X,
                                candidate.left_indices,
                                n_init=self.n_init,
                                random_state=self._seed(step + 1, cid * 2 + 1, 1),
                            ),
                            split_gain(
                                X,
                                candidate.right_indices,
                                n_init=self.n_init,
                                random_state=self._seed(step + 1, cid * 2 + 2, 1),
                            ),
                        )
                        self.kmeans_fit_calls_ += sum(
                            len(child_indices) >= 2
                            for child_indices in (
                                candidate.left_indices,
                                candidate.right_indices,
                            )
                        )
                        candidate_future = (
                            sum(child_gains)
                            if self.strategy == "alss_sum"
                            else max(child_gains)
                        )
                    candidate.future_gain = candidate_future
                    if lookahead_enabled:
                        lookahead_components[cid] = tuple(map(float, child_gains))
                    candidate_score = candidate.gain + self.lambda_ * candidate_future
                    if self.strategy in {"alss_relative", "alss_relative_score"}:
                        candidate_score = (
                            candidate_score / candidate.parent_sse
                            if candidate.parent_sse > np.finfo(float).eps
                            else 0.0
                        )
                    candidate_scores[cid] = candidate_score
                    lookahead_gains[cid] = candidate_future
                    scored.append((candidate_score, cid, candidate_future))
                score, selected_id, future_gain = min(
                    scored, key=lambda item: (-item[0], item[1])
                )
                proposal = proposals[selected_id]


            if self.strategy in {"largest", "inertia", "mean_sse", "random"}:
                for cid, members in eligible.items():
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(members.size),
                            "score_basis": self.strategy,
                            "selection_score": candidate_scores[cid],
                            "immediate_gain": proposal.gain if cid == selected_id else None,
                            "future_gain": None,
                            "candidate_evaluated": True,
                            "lookahead_evaluated": False,
                            "selected": cid == selected_id,
                            "top_r": None,
                        }
                    )
            elif self.strategy == "savaresi":
                for cid, members in eligible.items():
                    candidate = proposals.get(cid)
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(members.size),
                            "score_basis": "savaresi_shape_index",
                            "selection_score": candidate_scores.get(cid),
                            "immediate_gain": None if candidate is None else float(candidate.gain),
                            "future_gain": None,
                            "candidate_evaluated": candidate is not None,
                            "lookahead_evaluated": False,
                            "selected": cid == selected_id,
                            "top_r": 10,
                        }
                    )
            else:
                for cid, candidate in proposals.items():
                    is_lookahead = self.strategy in {
                        "alss",
                        "alss_relative",
                        "alss_sum",
                        "alss_relative_screen",
                        "alss_relative_score",
                    }
                    in_shortlist = cid in candidate_scores if is_lookahead else False
                    score_basis = (
                        "relative_gain_plus_lookahead"
                        if self.strategy == "alss_relative"
                        else "relative_screen_absolute_lookahead"
                        if self.strategy == "alss_relative_screen"
                        else "absolute_screen_relative_lookahead"
                        if self.strategy == "alss_relative_score"
                        else "gain_plus_sum_lookahead"
                        if self.strategy == "alss_sum"
                        else "gain_plus_lookahead"
                        if self.strategy == "alss"
                        else "relative_immediate_gain"
                        if self.strategy == "gain_relative"
                        else "gain"
                    )
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(candidate.members.size),
                            "score_basis": score_basis,
                            "selection_score": candidate_scores.get(cid),
                            "immediate_gain": float(candidate.gain),
                            "future_gain": lookahead_gains.get(cid),
                            "left_future_gain": (
                                lookahead_components[cid][0]
                                if cid in lookahead_components
                                else None
                            ),
                            "right_future_gain": (
                                lookahead_components[cid][1]
                                if cid in lookahead_components
                                else None
                            ),
                            "candidate_evaluated": True,
                            "lookahead_evaluated": in_shortlist and lookahead_enabled,
                            "lookahead_disabled_by_budget": (
                                in_shortlist
                                and self.horizon_aware
                                and self.lambda_ > 0
                                and step == self.n_clusters - 2
                            ),
                            "selected": cid == selected_id,
                            "top_r": self.top_r if is_lookahead else None,
                        }
                    )

            left_id, right_id = next_cluster_id, next_cluster_id + 1
            next_cluster_id += 2
            del clusters[selected_id]
            clusters[left_id] = proposal.left_indices
            clusters[right_id] = proposal.right_indices
            current_sse += proposal.child_sse - proposal.parent_sse

            self.trajectory_.append(
                {
                    "step": step + 1,
                    "strategy": self.strategy,
                    "split_cluster_id": selected_id,
                    "cluster_size": int(proposal.members.size),
                    "left_cluster_id": left_id,
                    "right_cluster_id": right_id,
                    "left_size": int(proposal.left_indices.size),
                    "right_size": int(proposal.right_indices.size),
                    "parent_sse": float(proposal.parent_sse),
                    "children_sse": float(proposal.child_sse),
                    "immediate_gain": float(proposal.gain),
                    "future_gain": future_gain,
                    "selection_score": None if score is None else float(score),
                    "lambda": self.lambda_ if self.strategy in {
                        "alss", "alss_relative", "alss_sum",
                        "alss_relative_screen", "alss_relative_score",
                    } else None,
                    "top_r": self.top_r if self.strategy in {
                        "alss", "alss_relative", "alss_sum",
                        "alss_relative_screen", "alss_relative_score",
                    } else None,
                    "shortlist_size": shortlist_size,
                    "lookahead_horizon_aware": self.horizon_aware,
                    "lookahead_enabled": self.strategy
                    in {
                        "alss",
                        "alss_relative",
                        "alss_sum",
                        "alss_relative_screen",
                        "alss_relative_score",
                    }
                    and self.lambda_ > 0
                    and (not self.horizon_aware or step < self.n_clusters - 2),
                    "kmeans_fit_calls": self.kmeans_fit_calls_,
                    "total_sse_after_split": float(current_sse),
                }
            )
            self.sse_history_.append(float(current_sse))
            self.n_iter_ += 1

        self.cluster_indices_ = {
            cid: members.copy() for cid, members in sorted(clusters.items())
        }
        self.labels_ = np.empty(X.shape[0], dtype=np.int64)
        for label, members in enumerate(self.cluster_indices_.values()):
            self.labels_[members] = label
        self.sse_ = cluster_sse(X, self.labels_)
        self.runtime_ = perf_counter() - started
        self.n_features_in_ = X.shape[1]
        return self
