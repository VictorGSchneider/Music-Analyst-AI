#!/usr/bin/env python3
"""Classify Spotify lyrics into sentiment classes using a local LLM via Ollama.

The script reads the CSV dataset, calls a local language model using the Ollama
CLI to classify the sentiment of each lyric, and aggregates the counts of
"Positiva", "Neutra" and "Negativa" labels. A lightweight lexicon-based
fallback is provided to keep the pipeline functional when the Ollama executable
is not available during development or automated testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, Iterable, Optional

LABELS = ("Positiva", "Neutra", "Negativa")


def detect_ollama() -> Optional[str]:
    """Return the Ollama executable path if available."""
    from shutil import which

    return which("ollama")


@dataclass
class ClassificationConfig:
    model: str
    temperature: float = 0.0
    top_p: float = 0.8


class LyricClassifier:
    """Classify lyrics either via Ollama or a heuristic fallback."""

    def __init__(self, config: ClassificationConfig, cache: Optional[Dict[str, str]] = None, cache_path: Optional[Path] = None) -> None:
        self.config = config
        self.cache = cache or {}
        self.cache_path = cache_path
        self.ollama_exec = detect_ollama()

    def _prompt(self, artist: str, song: str, lyrics: str) -> str:
        base_prompt = textwrap.dedent(
            f"""
            Classifique a seguinte letra de música do Spotify como "Positiva", "Neutra" ou "Negativa".
            Responda apenas com uma dessas três palavras em português europeu.

            Artista: {artist}
            Música: {song}
            Letra:
            {lyrics}
            """
        ).strip()
        return base_prompt

    def _call_ollama(self, prompt: str) -> Optional[str]:
        if not self.ollama_exec:
            return None
        cmd = [self.ollama_exec, "run", self.config.model]
        if self.config.temperature is not None:
            cmd.extend(["-o", f"temperature={self.config.temperature}"])
        if self.config.top_p is not None:
            cmd.extend(["-o", f"top_p={self.config.top_p}"])
        cmd.append(prompt)
        try:
            result = run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            return None
        except CalledProcessError as exc:
            sys.stderr.write(f"[WARN] Ollama returned non-zero exit status: {exc}\n")
            sys.stderr.flush()
            return None
        response = result.stdout.strip()
        if not response:
            return None
        first_word = response.split()[0]
        return first_word

    def _fallback_classify(self, lyrics: str) -> str:
        positive_words = {
            "love",
            "happy",
            "smile",
            "joy",
            "bom",
            "feliz",
            "amor",
            "paz",
            "alegria",
        }
        negative_words = {
            "sad",
            "cry",
            "pain",
            "bad",
            "triste",
            "raiva",
            "solidão",
            "morte",
            "ódio",
        }
        text = lyrics.lower()
        score = 0
        for word in positive_words:
            if word in text:
                score += 1
        for word in negative_words:
            if word in text:
                score -= 1
        if score > 0:
            return "Positiva"
        if score < 0:
            return "Negativa"
        return "Neutra"

    def classify(self, artist: str, song: str, lyrics: str) -> str:
        normalized_lyrics = lyrics.strip()
        cache_key = hashlib.sha1(normalized_lyrics.encode("utf-8")).hexdigest()
        cached = self.cache.get(cache_key)
        if cached in LABELS:
            return cached

        prompt = self._prompt(artist, song, normalized_lyrics)
        label = self._call_ollama(prompt)
        if label not in LABELS:
            label = self._fallback_classify(normalized_lyrics)

        self.cache[cache_key] = label
        return label

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_counts(output_path: Path, counts: Dict[str, int]) -> None:
    output_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")


def write_counts_csv(output_path: Path, counts: Dict[str, int]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "count"])
        for label in LABELS:
            writer.writerow([label, counts.get(label, 0)])


def iter_dataset(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            yield row


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Caminho para o arquivo CSV de entrada.")
    parser.add_argument("--model", default="llama3", help="Nome do modelo Ollama a ser utilizado.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar o número de músicas processadas.")
    parser.add_argument("--output-json", type=Path, default=Path("output/classification_counts.json"), help="Arquivo JSON de saída.")
    parser.add_argument("--output-csv", type=Path, default=Path("output/classification_counts.csv"), help="Arquivo CSV de saída.")
    parser.add_argument("--cache", type=Path, default=Path("output/classification_cache.json"), help="Arquivo de cache para resultados de classificação.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura a ser usada no modelo.")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.8, help="Top-p a ser usado no modelo.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    dataset_path: Path = args.input
    if not dataset_path.exists():
        sys.stderr.write(f"Arquivo de entrada não encontrado: {dataset_path}\n")
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)

    cache_data = load_cache(args.cache)
    classifier = LyricClassifier(
        ClassificationConfig(model=args.model, temperature=args.temperature, top_p=args.top_p),
        cache=cache_data,
        cache_path=args.cache,
    )

    counts = {label: 0 for label in LABELS}
    processed = 0
    try:
        for row in iter_dataset(dataset_path, args.limit):
            artist = row.get("artist", "")
            song = row.get("song", "")
            lyrics = row.get("text", "")
            if not lyrics.strip():
                continue
            label = classifier.classify(artist, song, lyrics)
            counts[label] = counts.get(label, 0) + 1
            processed += 1
            if processed % 10 == 0:
                sys.stderr.write(f"Processadas {processed} músicas...\n")
                sys.stderr.flush()
    except KeyboardInterrupt:
        sys.stderr.write("Interrompido pelo utilizador. Salvando resultados parciais...\n")

    classifier.save_cache()
    write_counts(args.output_json, counts)
    write_counts_csv(args.output_csv, counts)

    sys.stdout.write(json.dumps({"processed": processed, "counts": counts}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
