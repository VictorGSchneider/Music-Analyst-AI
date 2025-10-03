# Music-Analyst-AI

Aplicação paralela utilizando MPI em C para analisar o conjunto de dados [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset). O projeto calcula contagem de palavras das letras, artistas com mais músicas e realiza classificação de sentimento (Positiva, Neutra, Negativa) integrando com um modelo local de linguagem ou com um classificador léxico de fallback.

## Estrutura do projeto

```
├── Makefile
├── sentiment_classifier.py
├── spotify_millsongdata.csv
├── src
│   └── spotify_mpi.c
└── third_party
    └── uthash.h
```

- `src/spotify_mpi.c`: código C com MPI responsável pelo processamento paralelo das letras.
- `sentiment_classifier.py`: script auxiliar em Python que integra com um modelo local (Ollama) ou executa uma classificação baseada em léxico caso o modelo não esteja disponível.
- `third_party/uthash.h`: dependência externa usada para implementar tabelas hash em C.

## Pré-requisitos

- [MPI](https://www.mpi-forum.org/) com `mpicc` e `mpirun` instalados (por exemplo, via OpenMPI ou MPICH).
- Python 3.8+ com a biblioteca padrão.
- Opcional: [Ollama](https://ollama.com) ou outro servidor de modelo local acessível via comando `ollama run` para classificação neural. Caso não esteja disponível, o script em Python utilizará automaticamente o classificador léxico incluso.

## Compilação

```bash
make
```

O comando gera o executável `spotify_mpi` na raiz do projeto.

## Execução

O programa espera o arquivo `spotify_millsongdata.csv` na raiz. Para executar com 4 processos MPI:

```bash
mpirun -np 4 ./spotify_mpi
```

Também é possível informar um caminho alternativo para o CSV:

```bash
mpirun -np 4 ./spotify_mpi /caminho/para/spotify_millsongdata.csv
```

### Saída

O processo 0 imprime:

1. Top 20 palavras mais frequentes nas letras.
2. Top 20 artistas com maior quantidade de músicas.
3. Contagem total de letras classificadas como Positivas, Neutras e Negativas.
4. Métricas de desempenho: tempo médio e máximo de processamento dos dados e de classificação das letras.

Durante a execução cada processo gera um arquivo temporário `classification_rank_<id>.txt` com as letras que processou. Esse arquivo é removido automaticamente após a classificação.

## Classificador de sentimentos

O script `sentiment_classifier.py` recebe como entrada um arquivo com uma letra por linha e retorna três contagens (positivas, neutras, negativas). O programa MPI chama esse script automaticamente para cada partição de dados.

Uso isolado:

```bash
python3 sentiment_classifier.py --input classification_rank_0.txt --model llama3
```

Argumentos principais:

- `--input`: arquivo com uma letra por linha (obrigatório).
- `--model`: nome do modelo Ollama a ser utilizado. Se omitido ou se o comando `ollama run` falhar, o script aplica um classificador léxico local.
- `--report`: caminho opcional para salvar um relatório JSON com as contagens.

## Métricas de desempenho

O código utiliza `MPI_Wtime` para medir tempos:

- **Tempo de processamento**: leitura/distribuição das músicas, contagem de palavras e artistas.
- **Tempo de classificação**: execução do script Python para a partição local.

As métricas são agregadas usando `MPI_Reduce`, permitindo analisar gargalos (processos mais lentos) e impacto da classificação externa.

## Limpeza

```bash
make clean
```

Remove o executável gerado.

## Observações

- O classificador léxico não depende de rede e serve como fallback em ambientes sem Ollama.
- Para rodar com um modelo local real, instale o Ollama, baixe o modelo desejado (`ollama pull <modelo>`) e informe `--model` via variável de ambiente `OLLAMA_MODEL` ou ajustando o script conforme necessário.
