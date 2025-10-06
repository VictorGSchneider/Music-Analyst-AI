#!/usr/bin/env python3
"""Classifica o sentimento das letras do Spotify usando um LLM local via Ollama.

O script lê o dataset, envia cada letra para o modelo configurado e registra o
total de previsões "Positive", "Neutral" e "Negative". O modo ``--mock`` evita
chamadas reais ao modelo e utiliza uma heurística simples baseada em palavras,
útil para testes automatizados ou quando o servidor Ollama não está acessível.

Exemplo de uso::

    python scripts/sentiment_classifier.py spotify_millsongdata.csv \
        --model llama3 --limit 100 --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

try:
    import requests
except ImportError as exc:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


PROMPT_TEMPLATE = """You are an expert music analyst. Classify the overall sentiment of the following song lyrics as one of the following labels: Positive, Neutral, or Negative. Respond using only the label name with no explanations.\n\nLyrics:\n{lyrics}\n"""

DEFAULT_MODEL = "llama3"
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
SUPPORTED_LABELS = ("Positive", "Neutral", "Negative")


@dataclass
class ClassificationResult:
    label: str
    latency: float


class SentimentClassifier:
    """Encapsula a lógica de classificação de sentimento (real ou simulada)."""

    def __init__(self, model: str, mock: bool = False) -> None:
        """Configura o classificador indicando o modelo e se deve operar em modo mock."""
        self.model = model
        self.mock = mock
        if not mock and requests is None:
            raise RuntimeError(
                "The 'requests' package is required for live classification. Install it or use --mock."
            )

    def classify(self, lyrics: str) -> ClassificationResult:
        """Retorna a classe prevista para a letra fornecida."""
        lyrics = lyrics.strip()
        if not lyrics:
            return ClassificationResult("Neutral", 0.0)
        if self.mock:
            return self._mock_classify(lyrics)
        return self._ollama_classify(lyrics)

    def _mock_classify(self, lyrics: str) -> ClassificationResult:
        """Aplica uma heurística simples baseada em palavras positivas e negativas."""
        lowered = lyrics.lower()
        score = 0
        positive_keywords = ("love", "happy", "joy", "sunshine", "smile")
        negative_keywords = ("cry", "sad", "pain", "lonely", "tears")
        for word in positive_keywords:
            if word in lowered:
                score += 1
        for word in negative_keywords:
            if word in lowered:
                score -= 1
        label = "Neutral"
        if score > 0:
            label = "Positive"
        elif score < 0:
            label = "Negative"
        return ClassificationResult(label, 0.0)

    def _ollama_classify(self, lyrics: str) -> ClassificationResult:
        """Realiza a chamada HTTP ao servidor Ollama para obter a predição real."""
        assert requests is not None  # for type checkers
        payload = {
            "model": self.model,
            "prompt": PROMPT_TEMPLATE.format(lyrics=lyrics[:4000]),
            "stream": False,
        }
        start = time.perf_counter()
        response = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        raw_output = data.get("response", "").strip()
        label = self._normalise_label(raw_output)
        return ClassificationResult(label, elapsed)

    @staticmethod
    def _normalise_label(output: str) -> str:
        """Normaliza a resposta do modelo garantindo que o rótulo seja suportado."""
        cleaned = output.split()[0].strip().title()
        if cleaned not in SUPPORTED_LABELS:
            return "Neutral"
        return cleaned


def iter_lyrics(path: str, limit: int | None = None) -> Iterable[Tuple[str, str, str]]:
    """Itera sobre o CSV retornando artista, música e letra (até o limite opcional)."""
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row.get("artist", ""), row.get("song", ""), row.get("text", "")


def ensure_output_dir(path: str) -> None:
    """Garante que o diretório de saída exista antes de salvar os artefatos."""
    os.makedirs(path, exist_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    """Ponto de entrada do script para classificação de sentimentos."""
    parser = argparse.ArgumentParser(
        description="Classifica o sentimento das letras do Spotify com um LLM local"
    )
    parser.add_argument("dataset", help="Path to the spotify_millsongdata.csv dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of songs to classify")
    parser.add_argument("--output-dir", default="output", help="Directory where results are stored")
    parser.add_argument("--mock", action="store_true", help="Use a simple keyword heuristic instead of calling the LLM")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ensure_output_dir(args.output_dir)

    classifier = SentimentClassifier(args.model, mock=args.mock)
    counts: Dict[str, int] = {label: 0 for label in SUPPORTED_LABELS}
    per_song_rows = []

    for artist, song, lyrics in iter_lyrics(args.dataset, args.limit):
        result = classifier.classify(lyrics)
        counts[result.label] += 1
        per_song_rows.append(
            {
                "artist": artist,
                "song": song,
                "label": result.label,
                "latency_seconds": f"{result.latency:.4f}",
            }
        )

    aggregated_path = os.path.join(args.output_dir, "sentiment_totals.json")
    with open(aggregated_path, "w", encoding="utf-8") as fp:
        json.dump(counts, fp, indent=2)

    detailed_path = os.path.join(args.output_dir, "sentiment_details.csv")
    with open(detailed_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["artist", "song", "label", "latency_seconds"])
        writer.writeheader()
        writer.writerows(per_song_rows)

    print("Sentiment summary:")
    for label in SUPPORTED_LABELS:
        print(f"  {label}: {counts[label]}")
    print(f"Detailed results -> {detailed_path}")
    print(f"Aggregated counts -> {aggregated_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top level error reporting
        print(f"Error: {exc}", file=sys.stderr)
        raise
