# Análise de Desempenho

Este documento descreve como medir e interpretar as métricas de desempenho da aplicação MPI.

## Métricas coletadas automaticamente

Ao término da execução, o processo de *rank* 0 imprime:

- **Tempo de processamento**: tempo decorrido desde o início até a conclusão do processamento local (antes do envio/recebimento das tabelas de contagem). São apresentados os valores máximo, mínimo e médio entre todos os processos.
- **Tempo total**: tempo completo de execução em cada processo (incluindo sincronização e envio/recebimento de resultados). Também são exibidos máximo, mínimo e média.

Essas métricas são úteis para identificar desequilíbrios na distribuição de trabalho, gargalos de comunicação ou contendas de E/S.

## Recomendações para experimentos

1. **Variação do número de processos**: execute o programa com diferentes valores de `-np` (por exemplo, 1, 2, 4, 8, 16) e compare as métricas. O ganho ideal deve apresentar redução no tempo máximo de processamento.
2. **Amostragem do dataset**: use subconjuntos do CSV (por exemplo, `head -n 100000 spotify_millsongdata.csv > sample.csv`) para testar rapidamente e observar como o tempo cresce em relação ao volume de dados.
3. **I/O vs CPU**: monitore a utilização de CPU e disco (via `htop`, `iostat`, etc.) durante a execução. Leituras sequenciais do CSV podem se tornar o gargalo em discos lentos.
4. **Configuração de rede** (clusters): em ambientes distribuídos, avalie a latência e largura de banda da rede. A abordagem mestre-trabalhador usada aqui depende da comunicação ponto a ponto entre o *rank* 0 e os demais.

## Possíveis otimizações futuras

- **Leitura paralela** via MPI-IO para reduzir o gargalo do *rank* 0.
- **Compactação/serialização binária** das tabelas de contagem para diminuir o volume de dados transmitidos.
- **Balanceamento adaptativo**: permitir que processos solicitem novas linhas quando concluírem suas filas (modelo *work stealing*).
- **Pré-processamento do CSV** para normalizar letras e artistas, reduzindo trabalho de sanitização.

Documente sempre as condições de hardware, tamanho da amostra e número de processos para comparar execuções de forma justa.
