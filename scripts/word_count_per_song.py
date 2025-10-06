#!/usr/bin/env python3
"""Ferramenta auxiliar para contagem de palavras por música.

Este utilitário lê o dataset original do Spotify, tokeniza as letras mantendo
apóstrofos (para conservar contrações) e registra dois arquivos CSV:

1. ``word_counts_global.csv`` – frequência total de cada palavra.
2. ``word_counts_by_song.csv`` – frequência de cada palavra por artista e música.

O processamento ocorre de forma independente do executável MPI para não
interferir na coleta de métricas paralelas. Use quando precisar de uma visão
serial e detalhada das letras.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_REGEX = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ']+", re.UNICODE)


def tokenize(text: str) -> Iterable[str]:
    """Gera tokens com pelo menos 3 caracteres, preservando apóstrofos."""
    for match in TOKEN_REGEX.finditer(text):
        token = match.group().lower()
        if len(token) < 3:
            continue
        # Evita registrar strings compostas apenas por apóstrofos
        if not any(ch.isalnum() for ch in token):
            continue
        yield token


def detect_delimiter(sample: str) -> str:
    """Detecta o delimitador mais provável utilizando ``csv.Sniffer``."""
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except csv.Error:
        return ","


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conta palavras globalmente e por música de forma independente do MPI.",
    )
    parser.add_argument("csv_path", help="Caminho para o arquivo spotify_millsongdata.csv")
    parser.add_argument(
        "--output-dir",
        default="output/serial_word_counts",
        help="Diretório onde os resultados serão gravados (padrão: output/serial_word_counts)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Codificação do CSV de entrada (padrão: utf-8-sig)",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Delimitador do CSV (detectado automaticamente se omitido)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "word_counts_global.csv"
    per_song_path = output_dir / "word_counts_by_song.csv"

    with open(csv_path, "r", encoding=args.encoding, newline="") as fh:
        sample = fh.read(65536)
        fh.seek(0)
        delimiter = args.delimiter or detect_delimiter(sample)
        reader = csv.DictReader(fh, delimiter=delimiter)
        required_columns = {"artist", "song", "text"}
        if not required_columns.issubset(reader.fieldnames or {}):
            raise SystemExit(
                "CSV sem colunas esperadas. Campos necessários: artist, song, text."
            )

        global_counter: Counter[str] = Counter()
        total_rows = 0

        with open(per_song_path, "w", encoding="utf-8", newline="") as per_song_fh:
            per_song_writer = csv.writer(per_song_fh)
            per_song_writer.writerow(["artist", "song", "word", "count"])

            for row in reader:
                total_rows += 1
                artist = row.get("artist", "").strip()
                song = row.get("song", "").strip()
                text = row.get("text", "")
                word_counter: Counter[str] = Counter(tokenize(text))
                if not word_counter:
                    continue
                for word, count in word_counter.items():
                    global_counter[word] += count
                    per_song_writer.writerow([artist, song, word, count])

    with open(global_path, "w", encoding="utf-8", newline="") as global_fh:
        writer = csv.writer(global_fh)
        writer.writerow(["word", "count"])
        for word, count in global_counter.most_common():
            writer.writerow([word, count])

    print(
        "Concluído. Processadas",
        total_rows,
        "linhas. Arquivos gerados em",
        os.fspath(output_dir),
    )
    print(" -", os.fspath(global_path))
    print(" -", os.fspath(per_song_path))


if __name__ == "__main__":
    main()
