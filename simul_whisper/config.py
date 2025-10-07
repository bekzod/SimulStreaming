# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SimulWhisperConfig:
    '''Options that are common for all simul policies that could be implemented in SimulWhisper.'''
    model_path: str
    language: str = field(default="zh")
    nonspeech_prob: float = 1.0
    audio_min_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)

    # External LM (KenLM) shallow fusion
    kenlm_path: str = field(default=None, metadata={"help": "Path to KenLM .arpa or binary model"})
    lm_weight: float = field(default=0.0, metadata={"help": "Weight for LM shallow fusion (0 disables)"})
    lm_lowercase: bool = field(default=False, metadata={"help": "Lowercase text before LM scoring"})
    lm_length_weight: float = field(
        default=0.0,
        metadata={"help": "Length normalization weight (β in fusion score)"},
    )
    lm_length_exponent: float = field(
        default=1.0,
        metadata={"help": "Exponent applied to effective length in length normalization"},
    )
    lm_token_mode: Literal["auto", "word", "subword"] = field(
        default="auto",
        metadata={
            "help": "How to tokenize text for KenLM scoring: auto, word, or subword"
        },
    )
    lm_fusion_mode: Literal["online", "rescore", "both"] = field(
        default="online",
        metadata={
            "help": "When to apply LM weights: online (during beam), rescore (after), or both"
        },
    )
    lm_cache_size: int = field(
        default=8192,
        metadata={"help": "Max cached KenLM prefix states for incremental scoring"},
    )

    logdir: str = field(default="logdir", metadata={"help": "Directory to save audio segments and tokens for debugging purposes."})

@dataclass
class AlignAttConfig(SimulWhisperConfig):
    '''Options specific to the AlignAtt policy.'''
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200 # in frames. Max value is 1500. Higher value turns rewinds off.
    audio_max_len: float = 30.0
    cif_ckpt_path: str = ""
    never_fire: bool = False
