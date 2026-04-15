import numpy as np
from onnxruntime import InferenceSession


class OnnxPolicy:
    """Lightweight inference wrapper around an ONNX model.

    Drop-in replacement for MaskablePPO at inference time.
    Inputs match the observation dict produced by build_observation():
      grid    — np.float32 (9, 9, 1)
      scalars — np.float32 (258,)
    """

    def __init__(self, path: str):
        self.session = InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, obs, action_masks=None, deterministic=True):
        grid = obs["grid"][np.newaxis].astype(np.float32)        # (1, 9, 9, 1)
        scalars = obs["scalars"][np.newaxis].astype(np.float32)  # (1, 258)
        logits = self.session.run(
            ["logits"], {"grid": grid, "scalars": scalars}
        )[0][0]  # shape (6,)
        if action_masks is not None:
            logits[~action_masks.astype(bool)] = -np.inf
        if deterministic:
            return int(np.argmax(logits)), None
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs)), None
