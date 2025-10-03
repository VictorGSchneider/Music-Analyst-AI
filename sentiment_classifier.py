#!/usr/bin/env python3
"""Simple sentiment classifier integrating with a local LLM when available.

The script expects an input file containing one lyric per line and outputs three
integers separated by spaces representing the number of positive, neutral and
negative lyrics respectively. The integration attempts to use a locally running
Ollama model (if installed) and falls back to a rule-based classifier when the
model is not available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Basic sentiment lexicon as a fallback when no LLM is available.
POSITIVE_WORDS = {
    "amor",
    "feliz",
    "alegria",
    "bom",
    "boa",
    "fantástico",
    "incrível",
    "lindo",
    "maravilhoso",
    "sucesso",
    "peace",
    "love",
    "happy",
    "joy",
    "smile",
    "sunshine",
}

NEGATIVE_WORDS = {
    "triste",
    "odio",
    "ódio",
    "dor",
    "choro",
    "medo",
    "raiva",
    "solidão",
    "broken",
    "cry",
    "sad",
    "pain",
    "hurt",
    "dark",
    "lonely",
}

WORD_RE = re.compile(r"[\w']+")


@dataclass
class SentimentCounts:
    positive: int = 0
    neutral: int = 0
    negative: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return self.positive, self.neutral, self.negative


def call_ollama(prompt: str, model: str) -> Optional[str]:
    """Call a local Ollama model if available."""
    try:
        completed = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None
    except subprocess.TimeoutExpired:
        return None

    response = completed.stdout.strip()
    if not response:
        return None
    return response


def classify_with_llm(lyric: str, model: str) -> Optional[str]:
    prompt = (
        "Classifique a seguinte letra de música como Positiva, Neutra ou Negativa.\n"
        "Responda somente com uma das palavras: Positiva, Neutra ou Negativa.\n"
        f"Letra:\n{lyric.strip()}"
    )
    return call_ollama(prompt, model)


def classify_rule_based(lyric: str) -> str:
    words = WORD_RE.findall(lyric.lower())
    if not words:
        return "Neutra"
    pos_hits = sum(1 for token in words if token in POSITIVE_WORDS)
    neg_hits = sum(1 for token in words if token in NEGATIVE_WORDS)
    if pos_hits > neg_hits:
        return "Positiva"
    if neg_hits > pos_hits:
        return "Negativa"
    return "Neutra"


def classify_lyric(lyric: str, model: Optional[str]) -> str:
    if model:
        response = classify_with_llm(lyric, model)
        if response:
            normalized = response.strip().split()[0].capitalize()
            if normalized in {"Positiva", "Neutra", "Negativa"}:
                return normalized
    return classify_rule_based(lyric)


def load_lyrics(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            yield line.strip()


def aggregate_sentiments(lyrics: Iterable[str], model: Optional[str]) -> SentimentCounts:
    counts = SentimentCounts()
    for lyric in lyrics:
        if not lyric:
            counts.neutral += 1
            continue
        sentiment = classify_lyric(lyric, model)
        if sentiment == "Positiva":
            counts.positive += 1
        elif sentiment == "Negativa":
            counts.negative += 1
        else:
            counts.neutral += 1
    return counts


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classificador de sentimentos para letras.")
    parser.add_argument("--input", required=True, help="Arquivo com uma letra por linha.")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL"),
        help=(
            "Nome do modelo Ollama a ser utilizado. Se não informado ou em caso de erro, "
            "será utilizado o classificador baseado em léxico."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Arquivo JSON opcional para salvar o relatório de contagens.",
    )
    return parser.parse_args(args)


def save_report(report_path: str, counts: SentimentCounts) -> None:
    payload = {
        "positive": counts.positive,
        "neutral": counts.neutral,
        "negative": counts.negative,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(argv: List[str]) -> int:
    options = parse_args(argv)
    if not os.path.exists(options.input):
        print(f"Arquivo {options.input} não encontrado.", file=sys.stderr)
        return 1

    lyrics = load_lyrics(options.input)
    counts = aggregate_sentiments(lyrics, options.model)

    if options.report:
        save_report(options.report, counts)

    print(f"{counts.positive} {counts.neutral} {counts.negative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
