"""Backend base class."""
from ..config import Config


class Backend:
    name = "base"

    def __init__(self, cfg: Config = None, **kwargs):
        self.cfg = cfg

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

    def train_step(self, step):
        raise NotImplementedError

    def current_skill(self):
        return None

    def evaluate(self):
        return {}
