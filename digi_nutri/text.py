"""Frozen CLIP text encoder. Used to cache name embeddings, not at RGB test time."""

from __future__ import annotations

import gzip
import html
from pathlib import Path

import regex
import torch
import torch.nn.functional as F


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "clip_cache"


def _basic_clean(text: str) -> str:
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return " ".join(text.split())


def _bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(values, [chr(code) for code in codes]))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    found = set()
    previous = word[0]
    for char in word[1:]:
        found.add((previous, char))
        previous = char
    return found


class SimpleTokenizer:
    """CLIP BPE tokenizer. Ingredient names do not need the extra unicode fixer."""

    def __init__(self, bpe_path: Path) -> None:
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [token + "</w>" for token in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            rebuilt = []
            index = 0
            while index < len(word):
                try:
                    found = word.index(first, index)
                except ValueError:
                    rebuilt.extend(word[index:])
                    break
                rebuilt.extend(word[index:found])
                index = found
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    rebuilt.append(first + second)
                    index += 2
                else:
                    rebuilt.append(word[index])
                    index += 1
            word = tuple(rebuilt)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        text = " ".join(word)
        self.cache[token] = text
        return text

    def encode(self, text: str) -> list[int]:
        tokens = []
        cleaned = _whitespace_clean(_basic_clean(text)).lower()
        for token in regex.findall(self.pat, cleaned):
            mapped = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(mapped).split(" "))
        return tokens

    def tokenize(self, names: list[str], context_length: int = 77) -> torch.Tensor:
        sot = self.encoder["<|startoftext|>"]
        eot = self.encoder["<|endoftext|>"]
        rows = torch.zeros(len(names), context_length, dtype=torch.long)
        for index, name in enumerate(names):
            pieces = [sot, *self.encode(name), eot]
            if len(pieces) > context_length:
                pieces = pieces[:context_length]
                pieces[-1] = eot
            rows[index, : len(pieces)] = torch.tensor(pieces, dtype=torch.long)
        return rows


class FrozenCLIPTextEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.kind = "transformers"
        try:
            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.to(device)
        except Exception:
            weights = _cache_dir() / "ViT-B-32.pt"
            bpe = _cache_dir() / "bpe_simple_vocab_16e6.txt.gz"
            if not weights.is_file() or not bpe.is_file():
                raise FileNotFoundError(
                    "CLIP text weights are not available. Expected the official "
                    "ViT-B-32 checkpoint and BPE vocab under outputs/clip_cache/."
                )
            self.kind = "openai"
            self.simple = SimpleTokenizer(bpe)
            self.model = torch.jit.load(str(weights), map_location=device)
            self.model.eval()

    @torch.inference_mode()
    def encode(self, names: list[str]) -> torch.Tensor:
        if not names:
            return torch.zeros(0, 512)
        if self.kind == "transformers":
            tokens = self.tokenizer(names, padding=True, truncation=True, return_tensors="pt")
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            pooled = self.model(**tokens).pooler_output
            return F.normalize(pooled.float(), dim=-1)
        tokens = self.simple.tokenize(names).to(self.device)
        embedded = self.model.encode_text(tokens).float()
        return F.normalize(embedded, dim=-1)


def build_text_cache(
    names: list[str],
    model_name: str,
    device: str,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    unique = sorted(set(names))
    if not unique:
        return {}
    encoder = FrozenCLIPTextEncoder(model_name, device)
    rows = []
    for start in range(0, len(unique), batch_size):
        rows.append(encoder.encode(unique[start : start + batch_size]).cpu())
    matrix = torch.cat(rows, dim=0)
    return {name: matrix[index] for index, name in enumerate(unique)}


def save_text_cache(path: Path, cache: dict[str, torch.Tensor]) -> None:
    names = sorted(cache)
    matrix = torch.stack([cache[name].float().cpu() for name in names], dim=0)
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save({"names": names, "emb": matrix}, temporary)
    temporary.replace(path)


def load_text_cache(path: Path) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {name: blob["emb"][index] for index, name in enumerate(blob["names"])}
