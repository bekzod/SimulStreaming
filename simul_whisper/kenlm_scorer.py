import math
import os
from typing import Optional, List


class KenLMScorer:
    """Thin wrapper around kenlm to provide log-prob scoring.

    - Scores full text with optional lowercasing.
    - Returns natural-log probabilities compatible with model logprobs.
    """

    def __init__(self, model_path: str, lowercase: bool = False) -> None:
        try:
            import kenlm  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "kenlm package is not installed. Please install 'kenlm' to use LM fusion."
            ) from e

        resolved = self._resolve_model_path(model_path)
        self._kenlm = kenlm.Model(resolved)
        self._lower = lowercase

    @staticmethod
    def _log10_to_ln(x: float) -> float:
        return x * math.log(10.0)

    def score_text(self, text: str, add_bos: bool = True, add_eos: bool = False) -> float:
        if self._lower:
            text = text.lower()
        # kenlm.Model.score returns log10 probability
        log10_prob: float = self._kenlm.score(text, bos=add_bos, eos=add_eos)
        return self._log10_to_ln(log10_prob)

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolve a KenLM model path.

        Accepts:
        - Exact file paths (.bin/.klm/.binary/.arpa)
        - Basename without extension and tries known extensions in preferred order
        - Directory containing a single or multiple candidates; prefer binary formats
        """
        if os.path.isdir(path):
            prefs = [".bin", ".klm", ".binary", ".arpa"]
            candidates: List[str] = []
            for f in os.listdir(path):
                ext = os.path.splitext(f)[1].lower()
                if ext in prefs:
                    candidates.append(os.path.join(path, f))
            if not candidates:
                raise FileNotFoundError(f"No KenLM model found in directory: {path}")
            # sort by preference order
            def pref_key(p):
                ext = os.path.splitext(p)[1].lower()
                return prefs.index(ext), p
            candidates.sort(key=pref_key)
            return candidates[0]

        if os.path.exists(path):
            return path

        root, ext = os.path.splitext(path)
        if ext == "":
            for e in (".bin", ".klm", ".binary", ".arpa"):
                candidate = root + e
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(f"KenLM model not found: {path}")
