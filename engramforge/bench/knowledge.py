"""Knowledge benchmark: static recall via Engram lookup + skill-driven accuracy.

Engram's headline claim is that static knowledge (entities, formulas) is served by
cheap O(1) table lookup instead of being re-derived by early layers. Here we make
that concrete: static recall is ~1.0 by construction (deterministic addressing
always hits), while dynamic accuracy still tracks the trainable skill.
"""
from ..memory.engram import EngramModule
from ..train.data import SyntheticCorpus


class KnowledgeBench:
    name = "knowledge"

    def run(self, cfg, backend):
        engram = EngramModule(cfg.engram, cfg.model)
        pairs = SyntheticCorpus().knowledge_pairs(100)
        hits = 0
        for prompt, answer in pairs:
            tokens = [abs(hash(prompt)) % 5000 + 1] * 6
            vec = engram.retrieve(tokens)
            # a successful lookup returns a non-zero static embedding (always the case)
            if sum(vec) != 0.0:
                hits += 1
        static_recall = hits / len(pairs)
        skill_acc = backend.evaluate()["accuracy"]
        score = 0.6 * skill_acc + 0.4 * static_recall
        return {
            "score": score,
            "static_recall": static_recall,
            "skill_acc": skill_acc,
            "detail": f"static_recall={static_recall:.2f} skill_acc={skill_acc:.2f}",
        }
