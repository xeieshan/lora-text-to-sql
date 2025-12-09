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
        special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([format_full(e) for e in examples], trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")


def build_tiny_model(vocab_size: int, seed: int = 0):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg)
