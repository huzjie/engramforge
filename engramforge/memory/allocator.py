"""Sparsity allocation: the U-shaped trade-off between MoE and Engram.

Engram's first contribution formulates a capacity-allocation problem: given a
fixed parameter/compute budget, how much should go to neural computation (MoE)
vs static memory (Engram)? The per-domain loss as a function of the Engram
fraction is U-shaped -- too little memory wastes compute rebuilding static
patterns, too much memory starves reasoning -- and the optimal point shifts by
scale (hence a scaling law).
"""


class SparsityAllocator:
    """Fit and evaluate the U-shaped loss over the Engram capacity fraction."""

    def __init__(self):
        pass

    def loss(self, engram_ratio, optimal=0.35, curvature=2.0, baseline=0.5):
        """U-shaped loss: baseline + curvature * (ratio - optimal)^2."""
        d = engram_ratio - optimal
        return baseline + curvature * (d * d)

    def optimal_engram_ratio(self, compute_budget, param_budget):
        """Estimate the optimal Engram fraction at a given scale.

        The optimum drifts lower as compute budget grows (reasoning benefits more
        from compute at large scale), captured by a simple log-ratio term.
        """
        import math
        scale = math.log10(max(compute_budget, 1e3) / 1e3)
        # optimum starts ~0.42 at tiny scale and decays toward ~0.20 at huge scale
        optimal = max(0.18, 0.42 - 0.03 * scale)
        return optimal

    def sweep(self, n=21, optimal=0.35, curvature=2.0):
        pts = []
        for i in range(n + 1):
            r = i / n
            pts.append((r, self.loss(r, optimal, curvature)))
        return pts

    def best(self, n=201, optimal=0.35, curvature=2.0):
        r0, l0 = None, float("inf")
        for i in range(n + 1):
            r = i / n
            l = self.loss(r, optimal, curvature)
            if l < l0:
                r0, l0 = r, l
        return r0, l0

    def scaling_law(self, budgets):
        """Return (compute_budget, optimal_ratio) pairs for a list of budgets."""
        return [(b, self.optimal_engram_ratio(b, b)) for b in budgets]
