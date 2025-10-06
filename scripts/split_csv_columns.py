#!/usr/bin/env python3
"""
split_csv_columns.py
Divide um arquivo CSV em N arquivos, um por coluna, nomeando cada arquivo com o título da coluna.

Uso básico:
    python split_csv_columns.py caminho/para/dados.csv

Opções:
    --output-dir DIR        Diretório de saída (padrão: <nome_arquivo>_columns)
    --delimiter ";"         Delimitador (se não passar, tenta detectar automaticamente)
    --encoding "utf-8-sig"  Codificação (padrão: utf-8-sig)
    --no-header             Use se o CSV não tiver cabeçalho
    --force                 Sobrescreve arquivos existentes
    --quotechar "'"         Caractere de citação (padrão: ")
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path


def sanitize_filename(name: str, max_len: int = 80) -> str:
    name = (name or "").replace("\n", " ").replace("\r", " ").strip()
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)  # troca chars problemáticos
    name = re.sub(r"\s+", "_", name)                           # espaços -> _
    return (name or "col")[:max_len]


def detect_csv_params(
    f,
    sample_size: int = 65536,
    explicit_delimiter: str | None = None,
    quotechar: str = '"'
) -> dict:
    """Retorna kwargs para csv.reader/csv.writer (delimiter, quotechar, etc.)."""
    if explicit_delimiter:
        return dict(
            delimiter=explicit_delimiter,
            quotechar=quotechar,
            doublequote=True,
            skipinitialspace=False,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
    pos = f.tell()
    sample = f.read(sample_size)
    f.seek(pos)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dict(
            delimiter=dialect.delimiter,
            quotechar=(quotechar or '"'),
            doublequote=True,
            skipinitialspace=dialect.skipinitialspace,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
    except Exception:
        return dict(
            delimiter=",",
            quotechar=(quotechar or '"'),
            doublequote=True,
            skipinitialspace=False,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Divide um CSV em vários arquivos, um por coluna, usando o título como nome."
    )
    ap.add_argument("csv_path", help="Caminho do arquivo CSV de entrada")
    ap.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Diretório de saída",
    )
    ap.add_argument(
        "--delimiter",
        dest="delimiter",
        default=None,
        help="Delimitador do CSV (se omitido, detecta automaticamente)",
    )
    ap.add_argument(
        "--quotechar",
        dest="quotechar",
        default='"',
        help='Caractere de citação (padrão: ")',
    )
    ap.add_argument(
        "--encoding",
        dest="encoding",
        default="utf-8-sig",
        help="Codificação do arquivo (padrão: utf-8-sig)",
    )
    ap.add_argument(
        "--no-header",
        dest="no_header",
        action="store_true",
        help="Informe se o CSV NÃO possui cabeçalho",
    )
    ap.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Sobrescrever arquivos existentes",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.csv_path)
    if not in_path.exists():
        raise SystemExit(f"Erro: arquivo não encontrado: {in_path}")

    base_out = (
        Path(args.output_dir)
        if args.output_dir
        else in_path.with_suffix("").parent / f"{in_path.stem}_columns"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding=args.encoding, newline="") as f:
        fmt = detect_csv_params(
            f, explicit_delimiter=args.delimiter, quotechar=args.quotechar
        )
        reader = csv.reader(f, **fmt)

        try:
            first_row = next(reader)
        except StopIteration:
            raise SystemExit("CSV vazio.")

        if args.no_header:
            headers = [f"col{i+1}" for i in range(len(first_row))]
            first_data_row = first_row  # sem cabeçalho, já é dado
        else:
            headers = [
                (h if h is not None and str(h).strip() else f"col{i+1}")
                for i, h in enumerate(first_row)
            ]
            first_data_row = None

        num_cols = len(headers)

        # Gera nomes de arquivos baseados nos títulos (automático)
        seen_names: set[str] = set()
        filenames: list[str] = []
        for i, h in enumerate(headers, start=1):
            base_name = sanitize_filename(str(h))
            name = base_name or f"col{i}"
            # Evita colisões (títulos iguais): sufixa _2, _3, ...
            candidate = f"{name}.csv"
            k = 2
            while (
                candidate.lower() in seen_names
                or (base_out / candidate).exists()
                and not args.force
            ):
                candidate = f"{name}_{k}.csv"
                k += 1
            seen_names.add(candidate.lower())
            filenames.append(candidate)

        # Abre escritores
        files = []
        writers = []
        try:
            for i in range(num_cols):
                out_path = base_out / filenames[i]
                # Se existir e --force, sobrescreve; se não, já garantimos nome único acima
                fh = open(out_path, "w", encoding=args.encoding, newline="")
                writer = csv.writer(fh, **fmt)
                if not args.no_header:
                    writer.writerow([headers[i]])
                files.append(fh)
                writers.append(writer)

            # Se não havia cabeçalho, grava a primeira linha de dados
            if first_data_row is not None:
                for i in range(num_cols):
                    val = first_data_row[i] if i < len(first_data_row) else ""
                    writers[i].writerow([val])

            # Demais linhas
            for row in reader:
                for i in range(num_cols):
                    val = row[i] if i < len(row) else ""
                    writers[i].writerow([val])
        finally:
            for fh in files:
                try:
                    fh.close()
                except Exception:
                    pass

    print(f"Concluído. {num_cols} arquivo(s) gerado(s) em: {base_out}")
    for name in filenames:
        print(f" - {base_out / name}")


if __name__ == "__main__":
    main()
