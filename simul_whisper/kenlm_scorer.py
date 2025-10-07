import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from .whisper.tokenizer import Tokenizer

try:
    import kenlm  # type: ignore
except Exception as exc:  # pragma: no cover - kenlm is optional
    raise RuntimeError(
        "kenlm package is not installed. Please install 'kenlm' to use LM fusion."
    ) from exc


_DEFAULT_CACHE_LIMIT = 8192
_WORD_PATTERN = re.compile(r"\w+(?:['\-]\w+)*|[^\w\s]", re.UNICODE)
_TERMINAL_PUNCTUATION = set(".,;:!?)]}\"“”’‘-")


@dataclass
class _CacheEntry:
    """KenLM cached state for a token prefix."""

    state: "kenlm.State"
    log_prob: float
    pending: str
    decoded_text: str
    committed_units: int = 0


class KenLMScorer:
    """Stateful KenLM scorer with incremental prefix caching.

    Parameters
    ----------
    model_path:
        Path to .arpa or binary KenLM model.
    lowercase:
        Lowercase tokens before scoring.
    tokenizer:
        Optional Whisper tokenizer for subword→word buffering.
        Required when token_mode="auto" and no explicit mode is given.
    token_mode:
        "auto" (default), "word", or "subword". Word mode buffers text until a
        whitespace/punctuation boundary before scoring the LM; subword mode scores
        each decoded token immediately.
    cache_max_size:
        Maximum number of cached prefixes. Oldest entries are evicted first.
    """

    def __init__(
        self,
        model_path: str,
        *,
        lowercase: bool = False,
        tokenizer: Optional["Tokenizer"] = None,
        token_mode: str = "auto",
        cache_max_size: int = _DEFAULT_CACHE_LIMIT,
    ) -> None:
        resolved = self._resolve_model_path(model_path)
        self._kenlm = kenlm.Model(resolved)
        self._lower = lowercase
        self._tokenizer = tokenizer
        self._cache: "OrderedDict[Tuple[int, ...], _CacheEntry]" = OrderedDict()
        self._cache_limit = max(0, cache_max_size)
        if token_mode not in {"auto", "word", "subword"}:
            raise ValueError(f"Unsupported token_mode: {token_mode}")
        if token_mode == "auto":
            token_mode = "subword"
        if token_mode == "word" and tokenizer is None:
            raise ValueError(
                "Word-level token_mode requires a tokenizer to detokenize subwords."
            )
        self._token_mode = token_mode
        self._unk_token = "<unk>" if self._has_unk() else None

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _log10_to_ln(x: float) -> float:
        return x * math.log(10.0)

    def score_text(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> float:
        """Score full text in one go, returning ln probability."""
        text_for_scoring = text.lower() if self._lower else text
        log10_prob: float = self._kenlm.score(
            text_for_scoring, bos=add_bos, eos=add_eos
        )
        return self._log10_to_ln(log10_prob)

    # --------------------------------------------------------------------- #
    # Incremental state management
    # --------------------------------------------------------------------- #
    def start_entry(
        self, key: Tuple[int, ...], decoded_text: str, *, accumulate: bool = False
    ) -> _CacheEntry:
        """Create (or fetch) the cached LM entry for an initial prefix."""
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        entry = self._initial_entry()
        if decoded_text:
            entry = self._consume(
                entry,
                decoded_text,
                accumulate=accumulate,
                force_flush=False,
            )
        entry.decoded_text = decoded_text
        self._cache_put(key, entry)
        return entry

    def extend_entry(
        self,
        prefix_entry: _CacheEntry,
        key: Tuple[int, ...],
        decoded_text: str,
        *,
        accumulate: bool = True,
        force_flush: bool = False,
    ) -> _CacheEntry:
        """Advance LM state for `key` starting from ``prefix_entry``."""
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        appended = decoded_text[len(prefix_entry.decoded_text) :]
        if not appended and not force_flush:
            # Nothing new; reuse prefix.
            return prefix_entry

        entry = self._consume(
            prefix_entry,
            appended,
            accumulate=accumulate,
            force_flush=force_flush,
        )
        entry.decoded_text = decoded_text if appended else prefix_entry.decoded_text
        self._cache_put(key, entry)
        return entry

    def finalize_entry(
        self, entry: _CacheEntry, *, add_eos: bool = False
    ) -> _CacheEntry:
        """Flush any pending buffer (and optional EOS)."""
        final_entry = self._consume(entry, "", accumulate=True, force_flush=True)
        if add_eos:
            final_entry = self._score_unit(final_entry, "</s>")
        # We do not cache here because key is ambiguous at finalize time.
        return final_entry

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _initial_entry(self) -> _CacheEntry:
        state = self._new_state()
        self._kenlm.BeginSentenceWrite(state)
        return _CacheEntry(state=state, log_prob=0.0, pending="", decoded_text="")

    def _new_state(self) -> "kenlm.State":
        return kenlm.State()

    def _cache_get(self, key: Tuple[int, ...]) -> Optional[_CacheEntry]:
        if key in self._cache:
            # move to the end (most recently used)
            entry = self._cache.pop(key)
            self._cache[key] = entry
            return entry
        return None

    def _cache_put(self, key: Tuple[int, ...], entry: _CacheEntry) -> None:
        if self._cache_limit == 0:
            return
        self._cache[key] = entry
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def _consume(
        self,
        entry: _CacheEntry,
        appended_text: str,
        *,
        accumulate: bool,
        force_flush: bool,
    ) -> _CacheEntry:
        """Consume appended text and return a fresh cache entry."""
        text_buffer = entry.pending + appended_text
        if self._token_mode == "subword":
            units, leftover = self._extract_subword_units(text_buffer, force_flush)
        else:
            units, leftover = self._extract_word_units(text_buffer, force_flush)

        state = entry.state
        log_prob = entry.log_prob
        committed_units = entry.committed_units

        for unit in units:
            scored_entry = self._score_unit_raw(state, unit, accumulate)
            state = scored_entry.state
            if accumulate:
                log_prob += scored_entry.log_prob
            committed_units += 1

        return _CacheEntry(
            state=state,
            log_prob=log_prob,
            pending=leftover,
            decoded_text=entry.decoded_text + appended_text,
            committed_units=committed_units,
        )

    def _score_unit_raw(
        self, state: "kenlm.State", token: str, accumulate: bool
    ) -> _CacheEntry:
        """Score a single unit and return next state entry."""
        next_state = self._new_state()
        candidate = token.lower() if self._lower else token
        log10 = self._kenlm.BaseScore(state, candidate, next_state)
        if math.isinf(log10) and self._unk_token is not None:
            log10 = self._kenlm.BaseScore(state, self._unk_token, next_state)
        ln_prob = self._log10_to_ln(log10) if accumulate else 0.0
        return _CacheEntry(
            state=next_state,
            log_prob=ln_prob,
            pending="",
            decoded_text="",
        )

    def _score_unit(self, entry: _CacheEntry, token: str) -> _CacheEntry:
        scored = self._score_unit_raw(entry.state, token, True)
        return _CacheEntry(
            state=scored.state,
            log_prob=entry.log_prob + scored.log_prob,
            pending="",
            decoded_text=entry.decoded_text,
            committed_units=entry.committed_units + 1,
        )

    def _extract_subword_units(
        self, text: str, force: bool
    ) -> Tuple[List[str], str]:
        if not text:
            return [], ""
        # Subword mode treats each appended piece as a unit.
        units = [text] if text else []
        leftover = "" if force else ""
        return units, leftover

    def _extract_word_units(
        self, text: str, force: bool
    ) -> Tuple[List[str], str]:
        if not text:
            return [], ""
        matches = list(_WORD_PATTERN.finditer(text))
        if not matches:
            # No word characters yet; keep buffer (e.g., whitespace only).
            return [], text if not force else ""

        commit_count = len(matches)
        if not force and text and not text[-1].isspace():
            stripped = text.rstrip()
            if stripped and stripped[-1] not in _TERMINAL_PUNCTUATION:
                commit_count -= 1
        if commit_count <= 0:
            return [], text

        units: List[str] = []
        consume_upto = matches[commit_count - 1].end()
        for m in matches[:commit_count]:
            units.append(m.group())

        leftover = text[consume_upto:]
        # Avoid accumulating infinite whitespace.
        leftover = leftover.lstrip()
        return units, leftover

    def _has_unk(self) -> bool:
        try:
            prob = self._kenlm.score("<unk>", bos=False, eos=False)
            return not math.isinf(prob)
        except Exception:
            return False

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolve a KenLM model path."""
        if os.path.isdir(path):
            prefs = [".bin", ".klm", ".binary", ".arpa"]
            candidates: List[str] = []
            for fname in os.listdir(path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in prefs:
                    candidates.append(os.path.join(path, fname))
            if not candidates:
                raise FileNotFoundError(f"No KenLM model found in directory: {path}")

            def pref_key(candidate: str) -> Tuple[int, str]:
                ext = os.path.splitext(candidate)[1].lower()
                return prefs.index(ext), candidate

            candidates.sort(key=pref_key)
            return candidates[0]

        if os.path.exists(path):
            return path

        root, ext = os.path.splitext(path)
        if ext == "":
            for candidate_ext in (".bin", ".klm", ".binary", ".arpa"):
                candidate = root + candidate_ext
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(f"KenLM model not found: {path}")
