"""A tiny randomly initialised model and a locally trained tokenizer.

They let the whole pipeline (tokenise -> LoRA train -> save -> reload -> generate -> score)
run on a CPU in under a minute with no downloads, which is what CI does. The scores this
produces are meaningless; the point is that every step runs.
"""

from __future__ import annotations

from .data import Example
from .prompts import format_full


def build_tokenizer(examples: list[Example], vocab_size: int = 400):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
