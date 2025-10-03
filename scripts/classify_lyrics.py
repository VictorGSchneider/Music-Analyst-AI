#!/usr/bin/env python3
"""Classify Spotify lyrics using a local LLM via Ollama or a rule-based fallback."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Dict, Iterable, Tuple

SENTIMENT_CLASSES = ("Positiva", "Neutra", "Negativa")

PROMPT_TEMPLATE = """Classifique o sentimento geral da letra abaixo em uma única palavra: Positiva, Neutra ou Negativa.\nLetra:\n"""

FALLBACK_POSITIVE = {
    "love",
    "happy",
    "smile",
    "wonderful",
    "joy",
    "beautiful",
    "sunshine",
    "dance",
}
FALLBACK_NEGATIVE = {
    "sad",
    "cry",
    "lonely",
    "pain",
    "tears",
    "hate",
    "dark",
    "broken",
}


def call_ollama(model: str, prompt: str) -> str:
    """Call Ollama using subprocess and return the raw string output."""
    try:
        completed = subprocess.run(
            ["ollama", "run", model, prompt],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Ollama não foi encontrado no PATH. Instale o Ollama ou forneça --modo fallback."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Falha ao executar Ollama: {exc.stderr.strip() or exc}"
        ) from exc


def fallback_classify(text: str) -> str:
    text_lower = text.lower()
    score = 0
    for token in text_lower.split():
        token = ''.join(ch for ch in token if ch.isalpha())
        if token in FALLBACK_POSITIVE:
            score += 1
        elif token in FALLBACK_NEGATIVE:
            score -= 1
    if score > 0:
        return "Positiva"
    if score < 0:
        return "Negativa"
    return "Neutra"


def llm_classify(text: str, model: str) -> str:
    raw = call_ollama(model, PROMPT_TEMPLATE + text)
    cleaned = raw.strip().splitlines()[0].strip().lower()
    mapping = {
        "positiva": "Positiva",
        "positivo": "Positiva",
        "neutra": "Neutra",
        "negativa": "Negativa",
        "negativo": "Negativa",
    }
    return mapping.get(cleaned, "Neutra")


def classify_dataset(
    csv_path: str,
    limit: int | None,
    mode: str,
    model: str,
) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lyrics = row.get('text') or ''
            if mode == 'ollama':
                label = llm_classify(lyrics, model)
            else:
                label = fallback_classify(lyrics)
            counts[label] += 1
            total += 1
            if limit is not None and total >= limit:
                break
    return counts, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='Caminho para spotify_millsongdata.csv')
    parser.add_argument(
        '--modo',
        choices=('ollama', 'fallback'),
        default='fallback',
        help='Modo de classificação: usar Ollama (necessita instalação) ou heurística local.',
    )
    parser.add_argument(
        '--modelo',
        default='llama3',
        help='Nome do modelo Ollama a ser utilizado quando --modo=ollama.',
    )
    parser.add_argument(
        '--limite',
        type=int,
        default=None,
        help='Limite opcional de músicas a classificar (para testes rápidos).',
    )
    args = parser.parse_args()

    if args.modo == 'ollama':
        print(f'Classificando com Ollama e modelo {args.modelo}...')
    else:
        print('Classificando com heurística local (fallback).')

    try:
        counts, total = classify_dataset(args.csv, args.limite, args.modo, args.modelo)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f'Total de músicas classificadas: {total}')
    for label in SENTIMENT_CLASSES:
        print(f'{label}: {counts.get(label, 0)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
