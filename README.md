# Music-Analyst-AI

Aplicação paralela em C com MPI para processar o conjunto de dados
[Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset).
O objetivo é acelerar tarefas de análise exploratória das letras das músicas e
fornecer integração com um modelo local de linguagem para classificação de
sentimento.

## Funcionalidades

1. **Contagem de palavras** – cada processo MPI analisa uma partição do CSV e
   mantém contagens locais, que posteriormente são agregadas para produzir um
   ranking global das palavras mais frequentes.
2. **Artistas com mais músicas** – em paralelo, os processos também contam a
   quantidade de faixas por artista e reportam os artistas mais prolíficos.
3. **Classificação de sentimento** – script auxiliar em Python permite enviar
   as letras para um modelo local (por exemplo, via [Ollama](https://ollama.com))
   e sumarizar o total de músicas Positivas, Neutras e Negativas.
4. **Métricas de desempenho** – o executável MPI exporta métricas de tempo de
   processamento e comunicação para análise comparativa entre diferentes
   números de processos.

## Pré-requisitos

- Compilador e runtime MPI (`mpicc`, `mpirun`).
- Python 3.9+ para a etapa de classificação com LLM.
- (Opcional) Pacote `requests` para chamadas HTTP ao servidor do modelo.
- (Opcional) Servidor Ollama ou outra solução que exponha um endpoint
  compatível com `/api/generate`.

## Compilação

```bash
make
```

O binário principal será gerado em `bin/parallel_spotify`.

## Execução da análise paralela

```bash
mpirun -np <processos> ./bin/parallel_spotify spotify_millsongdata.csv \
  [--word-limit N] [--artist-limit N] [--output-dir diretório]
```

Parâmetros opcionais:

- `--word-limit`: limita o número de palavras escritas em `word_counts.csv`
  (valor padrão: 100, use `0` para salvar todas).
- `--artist-limit`: controla a quantidade de artistas exportados (padrão: 50).
- `--output-dir`: diretório onde os artefatos são gerados (padrão: `output`).

Ao final da execução são produzidos:

- `word_counts.csv` – ranking decrescente de palavras.
- `top_artists.csv` – artistas ordenados pela quantidade de músicas.
- `performance_metrics.json` – tempos mínimo, médio e máximo por processo.

## Classificação de sentimento com modelo local

O script Python `scripts/sentiment_classifier.py` consome o mesmo dataset e
consulta um modelo via HTTP. Ele pode operar em dois modos:

1. **Modo real** (padrão) – envia requisições para um servidor compatível com
   Ollama em `http://localhost:11434`. Ajuste a variável de ambiente
   `OLLAMA_ENDPOINT` ou utilize `--model` para trocar o modelo.
2. **Modo `--mock`** – aplica uma heurística simples baseada em palavras-chave
   para facilitar testes sem LLM instalado.

Exemplo de uso (modo real):

```bash
python scripts/sentiment_classifier.py spotify_millsongdata.csv \
  --model llama3 --limit 200 --output-dir output
```

Modo simulado:

```bash
python scripts/sentiment_classifier.py spotify_millsongdata.csv --mock
```

Saídas geradas:

- `sentiment_totals.json` – contagem agregada por classe.
- `sentiment_details.csv` – detalhamento por música e tempo de inferência.

## Medindo desempenho

O arquivo `performance_metrics.json` produzido pelo executável MPI contém
estatísticas de tempo médio, mínimo e máximo para cada estágio (cômputo local e
tempo total). Para facilitar a comparação entre diferentes tamanhos de cluster,
utilize o script `scripts/run_performance.sh`:

```bash
scripts/run_performance.sh spotify_millsongdata.csv 2 4 8
```

O script não recompila o projeto automaticamente; execute `make` antes de
realizar os experimentos para garantir que o binário esteja atualizado.

## Estrutura do repositório

```
├── bin/parallel_spotify        # executável (gerado pelo make)
├── output/                     # resultados (criado em tempo de execução)
├── scripts/
│   ├── run_performance.sh      # facilita execuções com vários processos
│   └── sentiment_classifier.py # classificação de sentimento com LLM local
└── src/parallel_spotify.c      # código-fonte principal em C/MPI
```

## Histórico de versões

| Versão | Descrição | Principais diferenças |
| ------ | --------- | --------------------- |
| v0.1 (`bdde614`) | Estrutura inicial do projeto. | Repositório criado com README mínimo e configuração base. |
| v0.2 (`4f78f8e`, `732a70c`) | Detalhamento do enunciado. | README atualizado com os requisitos completos do trabalho prático. |
| v0.3 (`293e8f8`) | Inclusão do dataset. | Arquivo `spotify_millsongdata.csv` adicionado para possibilitar testes locais. |
| v1.0 (`8cc21fe`) – **versão atual** | Implementação paralela com MPI e pipeline de sentimento. | Código C distribuído, geração de métricas de desempenho, scripts auxiliares em Python e shell, documentação completa em português. |

## Observações

- Os arquivos CSV gerados podem ser grandes; ajuste `--word-limit` e
  `--artist-limit` conforme necessário.
- Certifique-se de apontar o `mpirun` para o mesmo `mpicc` utilizado na
  compilação para evitar incompatibilidades de runtime.
