# Music-Analyst-AI

Aplicação para análise paralela do dataset [Spotify Million Song](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) utilizando MPI em C.
O projeto entrega três componentes principais:

1. **Contagem de palavras** nas letras das músicas.
2. **Ranking dos artistas** com maior número de músicas.
3. **Classificação de sentimento** (Positiva, Neutra ou Negativa) para cada letra através de integração com um LLM local (via [Ollama](https://ollama.com)) ou heurística de fallback.

Além disso, são disponibilizadas métricas de desempenho para avaliar o impacto do paralelismo.

## Estrutura do projeto

```
.
├── docs/
│   └── performance_analysis.md   # Guia para medições e análise de desempenho
├── include/
│   └── hashmap.h                 # Estruturas auxiliares de mapa hash
├── scripts/
│   └── classify_lyrics.py        # Classificação de sentimentos via LLM ou fallback
├── src/
│   ├── hashmap.c                 # Implementação da tabela hash
│   └── mpi_spotify.c             # Aplicação principal em MPI
├── Makefile                      # Processo de compilação com mpicc
├── README.md                     # Este arquivo
└── spotify_millsongdata.csv      # Dataset original (precisa estar presente)
```

## Pré-requisitos

- [MPI](https://www.open-mpi.org/) (por exemplo, OpenMPI) com `mpicc`, `mpirun` ou `mpiexec`.
- Compilador C compatível com C11.
- Python 3.9+ para o script de classificação.
- (Opcional) [Ollama](https://ollama.com) instalado localmente e configurado com um modelo adequado para sentimento (ex.: `llama3`).

## Compilação

```bash
mpicc -O2 -Wall -Wextra -std=c11 -Iinclude -o mpi_spotify src/hashmap.c src/mpi_spotify.c
```

ou simplesmente:

```bash
make
```

## Execução da análise paralela

```bash
mpirun -np <numero_processos> ./mpi_spotify spotify_millsongdata.csv
```

A aplicação distribui cada linha do CSV entre os processos MPI, efetua a contagem local e consolida os resultados no *rank* 0. O processo principal imprime:

- Top 20 palavras mais frequentes.
- Top 20 artistas com mais músicas.
- Métricas de tempo (mínimo, máximo e média) para processamento e tempo total por processo.

## Classificação de sentimento com LLM

O script `scripts/classify_lyrics.py` realiza a classificação de cada letra em três categorias.

### Uso com Ollama

```bash
python3 scripts/classify_lyrics.py spotify_millsongdata.csv --modo ollama --modelo llama3 --limite 1000
```

- `--limite` é opcional para restringir a quantidade de músicas processadas (útil para testes iniciais).
- Ajuste o nome do modelo conforme os modelos instalados no Ollama.

### Modo de fallback

Caso não haja um LLM disponível, utilize o modo `fallback`, que aplica uma heurística simples baseada em palavras-chave. Embora menos precisa, permite validar toda a tubulação do processo.

```bash
python3 scripts/classify_lyrics.py spotify_millsongdata.csv --modo fallback --limite 1000
```

O script mostra o total de músicas analisadas e a contagem por classe.

## Métricas de desempenho

O arquivo [`docs/performance_analysis.md`](docs/performance_analysis.md) descreve como coletar e interpretar as métricas emitidas pela aplicação MPI, além de sugerir abordagens para testes variando o número de processos, tamanho de amostras e características de hardware.

## Observações

- O dataset original contém cerca de um milhão de músicas. Para experimentos rápidos, considere criar subconjuntos menores (por exemplo, usando `head` ou ferramentas como `split`).
- Certifique-se de que o arquivo CSV esteja codificado em UTF-8 para evitar problemas de leitura.
- A classificação via LLM pode ser custosa; recomenda-se processar em lotes ou aplicar paralelismo adicional na camada Python caso necessário.
