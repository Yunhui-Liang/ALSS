"""Candidate split construction and split-gain helpers."""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .evaluation import sse_for_indices


@dataclass
class SplitProposal:
    cluster_id: int
    members: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray
    parent_sse: float
    child_sse: float
    gain: float
    future_gain: float | None = None


def propose_split(
    X: np.ndarray,
    members: np.ndarray,
    cluster_id: int,
    n_init: int = 10,
    random_state: int = 0,
) -> SplitProposal:
    """Fit a deterministic two-means split for one active cluster.

    If k-means returns fewer than two non-empty groups (possible for duplicate
    observations), split the cluster at its median along the highest-variance
    feature so BKM can still return the requested number of clusters.
    """
    members = np.asarray(members, dtype=np.int64)
    if members.size < 2:
        raise ValueError("A candidate cluster must contain at least two samples")
    subset = X[members]
    parent_sse = sse_for_indices(X, members)

    model = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        with threadpool_limits(limits=1):
            local_labels = model.fit_predict(subset)

    if np.unique(local_labels).size == 2:
        left_indices = members[local_labels == 0]
        right_indices = members[local_labels == 1]
    else:
        axis = int(np.argmax(np.var(subset, axis=0)))
        order = np.lexsort((members, subset[:, axis]))
        cut = members.size // 2
        left_indices, right_indices = members[order[:cut]], members[order[cut:]]

    child_sse = sse_for_indices(X, left_indices) + sse_for_indices(X, right_indices)
    gain = max(0.0, float(parent_sse - child_sse))
    return SplitProposal(
        cluster_id=int(cluster_id),
        members=members.copy(),
        left_indices=left_indices,
        right_indices=right_indices,
        parent_sse=float(parent_sse),
        child_sse=float(child_sse),
        gain=gain,
    )


def split_gain(
    X: np.ndarray,
    members: np.ndarray,
    n_init: int = 10,
    random_state: int = 0,
) -> float:
    """Return the observed one-step two-means SSE reduction."""
    members = np.asarray(members, dtype=np.int64)
    if members.size < 2:
        return 0.0
    return propose_split(
        X,
        members,
        cluster_id=-1,
        n_init=n_init,
        random_state=random_state,
    ).gain


def savaresi_shape_index(X: np.ndarray, proposal: SplitProposal) -> float | None:
    """Compute the projected shape score J from Savaresi et al. (2002).

    The K-means child-centroid difference defines a unit direction. Points are
    projected onto that direction and assigned to half-spaces through the
    parent centroid, as in the published definition. ``None`` denotes a
    degenerate split for which the score is undefined.
    """
    left_center = np.mean(X[proposal.left_indices], axis=0)
    right_center = np.mean(X[proposal.right_indices], axis=0)
    direction = right_center - left_center
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    direction /= norm

    points = X[proposal.members]
    parent_center = np.mean(points, axis=0)
    projections = (points - parent_center) @ direction
    left_projection = projections[projections <= 0.0]
    right_projection = projections[projections > 0.0]
    if left_projection.size == 0 or right_projection.size == 0:
        return None

    left_scale = float(np.min(left_projection))
    right_scale = float(np.max(right_projection))
    if abs(left_scale) <= np.finfo(float).eps or abs(right_scale) <= np.finfo(float).eps:
        return None

    left_normalized = left_projection / left_scale
    right_normalized = right_projection / right_scale
    left_mean = float(np.mean(left_normalized))
    right_mean = float(np.mean(right_normalized))
    left_variance = float(np.mean((left_normalized - left_mean) ** 2))
    right_variance = float(np.mean((right_normalized - right_mean) ** 2))

    mean_centroid_energy = 0.5 * (left_mean**2 + right_mean**2)
    mean_scatter = 0.5 * (left_variance + right_variance)
    if mean_centroid_energy <= np.finfo(float).eps:
        return None
    return mean_scatter / mean_centroid_energy
