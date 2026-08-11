"""CognitiveModes: Kuramoto phases as a detector of mental state.

THE INNOVATION. The Kuramoto oscillators aren't just a routing mechanism —
they're a DYNAMICAL SYSTEM whose phase pattern reflects the current "cognitive
mode" of the engine. This module:

    1. Extracts features from the phase vector (synchronization, clustering).
    2. Clusters phase patterns into cognitive modes (UNSUPERVISED — the modes
       emerge from the data, not from external labels).
    3. Lets the engine ADAPT its behavior based on its current mode.

This is what makes Fractus feel ALIVE — it has mental states that change how
it processes information, like a human shifting between focused work and
creative brainstorming.

UNSUPERVISED APPROACH (replaces the original supervised MLP):
    Instead of labelling phases with mode names (which is arbitrary), we collect
    phase features during a training run and cluster them with k-means. The
    clusters that emerge ARE the cognitive modes — defined by their centroids
    in the (synchronization, mean_phase, variance, sin/cos) feature space. Mode
    names are assigned a posteriori by interpreting the cluster characteristics
    (high sync = "focused", low sync = "exploratory", etc.).

Usage:
    modes = CognitiveModes(n_oscillators=8, n_modes=4)
    # Collect phases during training, then fit:
    modes.fit(phase_samples)  # phase_samples: (N_samples, n_oscillators)
    # Classify at runtime:
    mode = modes.classify(phases)  # → {"mode": "cluster_0", "confidence": 0.82, ...}
"""


import torch
import torch.nn as nn


class CognitiveModes(nn.Module):
    """Classify the Kuramoto phase state into cognitive modes via clustering.

    Modes are discovered unsupervised via k-means on phase features. No labels,
    no MLP — the clusters emerge from the structure of the phase space.

    Args:
        n_oscillators: number of Kuramoto oscillators.
        n_modes: number of modes (= k-means clusters).
        mode_names: optional names (assigned after fit by interpretation).
    """

    def __init__(
        self,
        n_oscillators: int = 8,
        n_modes: int = 4,
        mode_names: list = None,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.n_modes = n_modes
        if mode_names is None:
            mode_names = [f"mode_{i}" for i in range(n_modes)]
        self.mode_names = mode_names[:n_modes]
        self.n_features = 3 + 2 * n_oscillators

        # Centroids: learned via k-means during fit(). Stored as a buffer.
        self.register_buffer("centroids", torch.zeros(n_modes, self.n_features))
        self._fitted = False

    def extract_features(self, phases: torch.Tensor) -> torch.Tensor:
        """Extract cognitive features from the phase vector.

        Args:
            phases: (..., N) oscillator phases in [0, 2π).
        Returns:
            features: (..., 3 + 2*N) feature vector.
        """
        *leading, N = phases.shape
        phases_flat = phases.reshape(-1, N)  # (B, N)

        sin_p = torch.sin(phases_flat)
        cos_p = torch.cos(phases_flat)

        # Feature 1: order parameter r (synchronization degree).
        r = torch.sqrt(cos_p.mean(dim=-1) ** 2 + sin_p.mean(dim=-1) ** 2 + 1e-12)

        # Feature 2: mean phase.
        mean_phase = torch.atan2(sin_p.mean(dim=-1), cos_p.mean(dim=-1))

        # Feature 3: phase variance.
        phase_var = sin_p.var(dim=-1) + cos_p.var(dim=-1)

        # Features 4+: per-oscillator sin/cos.
        osc_features = torch.cat([sin_p, cos_p], dim=-1)  # (B, 2N)

        features = torch.cat([
            r.unsqueeze(-1),
            mean_phase.unsqueeze(-1),
            phase_var.unsqueeze(-1),
            osc_features,
        ], dim=-1)  # (B, 3 + 2N)

        return features.reshape(*leading, features.shape[-1])

    def fit(self, phase_samples: torch.Tensor, n_iters: int = 50) -> dict:
        """Fit k-means on collected phase samples (unsupervised).

        Args:
            phase_samples: (N_samples, n_oscillators) phases collected during training.
            n_iters: k-means iterations.
        Returns:
            dict with cluster info for interpretation.
        """
        features = self.extract_features(phase_samples)  # (N_samples, n_features)
        N = features.shape[0]
        K = self.n_modes

        if N < K:
            # Not enough samples — pad with noise.
            features = torch.cat([features, torch.randn(K - N, self.n_features)], dim=0)
            N = K

        # Initialize centroids: random samples.
        idx = torch.randperm(N)[:K]
        self.centroids = features[idx].clone()

        for _ in range(n_iters):
            # Assign each sample to nearest centroid (cosine distance).
            # Normalize for cosine.
            feat_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            cent_norm = self.centroids / (self.centroids.norm(dim=-1, keepdim=True) + 1e-8)
            sims = feat_norm @ cent_norm.T  # (N, K) cosine similarity
            assignments = sims.argmax(dim=-1)  # (N,)

            # Update centroids.
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    self.centroids[k] = features[mask].mean(dim=0)

        self._fitted = True

        # Compute cluster statistics for interpretation.
        cluster_info = {}
        for k in range(K):
            mask = assignments == k
            if mask.any():
                cluster_features = features[mask]
                cluster_info[k] = {
                    "size": mask.sum().item(),
                    "mean_sync": cluster_features[:, 0].mean().item(),  # r
                    "mean_var": cluster_features[:, 2].mean().item(),
                }
            else:
                cluster_info[k] = {"size": 0, "mean_sync": 0, "mean_var": 0}
        return cluster_info

    def classify(self, phases: torch.Tensor) -> dict:
        """Classify the current cognitive mode (nearest centroid).

        Args:
            phases: (N,) or (1, N) or (..., N) oscillator phases.
        Returns:
            dict with "mode" (str), "confidence" (float), and "all_modes" (dict).
        """
        if phases.dim() == 1:
            phases = phases.unsqueeze(0)
        features = self.extract_features(phases)  # (1, n_features)

        if not self._fitted:
            # Before fitting, return uniform.
            return {
                "mode": "unfitted",
                "confidence": 1.0 / self.n_modes,
                "all_modes": {name: 1.0 / self.n_modes for name in self.mode_names},
            }

        # Cosine similarity to each centroid.
        feat_norm = features[0] / (features[0].norm() + 1e-8)
        cent_norm = self.centroids / (self.centroids.norm(dim=-1, keepdim=True) + 1e-8)
        sims = cent_norm @ feat_norm  # (K,)
        probs = torch.softmax(sims * 5.0, dim=-1)  # temperature-scaled

        top_idx = probs.argmax(dim=-1).item()
        top_prob = probs[top_idx].item()
        mode_name = self.mode_names[top_idx] if top_idx < len(self.mode_names) else f"mode_{top_idx}"

        all_modes = {
            (self.mode_names[i] if i < len(self.mode_names) else f"mode_{i}"): probs[i].item()
            for i in range(self.n_modes)
        }

        return {
            "mode": mode_name,
            "confidence": top_prob,
            "all_modes": all_modes,
        }

    def label_modes(self, names: list):
        """Assign human-readable names to clusters after fitting (a posteriori).

        Args:
            names: list of n_modes names, in cluster order.
        """
        if len(names) != self.n_modes:
            raise ValueError(f"expected {self.n_modes} names, got {len(names)}")
        self.mode_names = names

    def info(self) -> dict:
        return {
            "n_oscillators": self.n_oscillators,
            "modes": self.mode_names,
            "n_features": self.n_features,
            "fitted": self._fitted,
        }
