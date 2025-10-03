# Music-Analyst-AI

Este projeto contém uma solução completa para analisar o dataset [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) com MPI em C e um pipeline auxiliar em Python para classificação de sentimento com um LLM local.

## Visão Geral

### Funcionalidades implementadas
- **Contagem de palavras**: distribuição das letras entre processos MPI, tokenização e agregação de frequências globais.
- **Artistas com mais músicas**: contagem paralela do número de músicas por artista.
- **Classificação de sentimento**: script em Python que integra com o [Ollama](https://ollama.com) para invocar um modelo local e classificar cada letra em `Positiva`, `Neutra` ou `Negativa`, com fallback heurístico para ambientes sem o Ollama.
- **Métricas de desempenho**: o binário MPI emite tempo de processamento por processo, quantidade de registros atribuídos e tempo total.

## Pré-requisitos
- Compilador MPI (ex.: OpenMPI, MPICH) com `mpicc` disponível.
- Python 3.9+ com as bibliotecas da stdlib (não há dependências extras) para o script de classificação.
- [Ollama](https://ollama.com) instalado e configurado com pelo menos um modelo de linguagem local (ex.: `llama3`) para executar a classificação real das letras.
- Dataset `spotify_millsongdata.csv` disponível no diretório raiz do projeto (já incluso neste repositório como exemplo).

## Compilação

```bash
make
```

O comando gera o binário `mpi_spotify` na raiz do projeto.

## Execução da análise MPI

```bash
mpirun -np <processos> ./mpi_spotify --input spotify_millsongdata.csv [opções]
```

Opções principais:

| Opção | Descrição |
| --- | --- |
| `--output-dir <dir>` | Diretório onde os resultados serão gravados (padrão: `output/`). |
| `--max-records <n>` | Limita a quantidade de músicas processadas (útil para testes rápidos). |
| `--top-words <n>` | Salva apenas os `n` termos mais frequentes (0 = todos). |
| `--top-artists <n>` | Salva apenas os `n` artistas com mais músicas (0 = todos). |

Saídas geradas no diretório especificado:

- `word_counts.csv`: lista ordenada por frequência das palavras encontradas nas letras.
- `artist_song_counts.csv`: artistas ordenados pela quantidade de músicas.
- Métricas de desempenho são exibidas no terminal no formato `rank, registros processados, tempo de processamento, tempo total`.

### Notas de desempenho
- O processo de rank 0 lê o CSV sequencialmente e distribui as músicas para os demais processos utilizando um esquema round-robin.
- Os tempos exibidos permitem identificar gargalos de I/O (tempo total maior que o tempo de processamento local) e desequilíbrios de carga.
- Use a opção `--max-records` para executar benchmarks controlados e comparar diferentes quantidades de processos.

## Classificação das letras com LLM local

O script `scripts/classify_lyrics.py` encapsula a integração com o Ollama. Exemplo de execução:

```bash
python scripts/classify_lyrics.py \
  --input spotify_millsongdata.csv \
  --model llama3 \
  --limit 1000 \
  --temperature 0.0 \
  --top-p 0.8 \
  --output-json output/classification_counts.json \
  --output-csv output/classification_counts.csv
```

Características do script:

- Detecção automática do executável `ollama`. Caso não esteja disponível, aplica um classificador heurístico simples baseado em léxico, permitindo validar o pipeline em ambientes sem GPU/modelo local.
- Cache incremental das classificações em `output/classification_cache.json`, evitando chamadas repetidas ao modelo em execuções subsequentes.
- Emissão de progresso a cada 10 músicas processadas.
- Saídas em JSON (para automação) e CSV (para uso em planilhas/BI).

> **Dica**: Para o dataset completo recomenda-se executar o script em lotes (via `--limit`) e manter o cache ligado; o tempo de inferência dependerá do modelo escolhido e do hardware local.

## Estrutura de diretórios

```
├── Makefile
├── README.md
├── scripts/
│   └── classify_lyrics.py
├── src/
│   └── mpi_spotify.c
└── spotify_millsongdata.csv
```

## Métricas adicionais

- Utilize `mpirun -np <n> ./mpi_spotify --max-records <m>` com diferentes valores de `<n>` para observar o impacto na distribuição de registros e no tempo total.
- Registre os resultados de tempo no terminal juntamente com o número de processos para montar gráficos de speedup/eficiência.
- Combine os relatórios de contagem (CSV) com os resultados do script de classificação para análises mais profundas, como correlações entre sentimentos predominantes e artistas com maior produção.

## Licença

Este projeto segue a licença MIT (ver arquivo `LICENSE`).
