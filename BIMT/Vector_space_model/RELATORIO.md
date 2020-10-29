# Relatório
Podemos encontrar todos estes arquivos na pasta _EVALUATION_.
## Métricas sem stemmer
1. Métricas do conjunto de consultas
    - Mean Average Precision: 0.2292
    - Mean Reciprocal Rank: 0.7810
    - Average Discounted Gain: 0.1755
    - Normalized Discounted Gain: 0.4371
    - O gráfico de 11 pontos de precisão e recall médio para o conjunto de consultas pode ser encontrado na pasta mencionada anteriormente com o nome de _avg_11points_no_stemmer.jpg_ e _11points_no_stemmer.csv_

2. Métricas de consulta individuais
    -   O arquivo para análise de F1, P@5, P@10, 
    MAP para cada consulta e R-precision podem ser encontrados no diretório mencionado no inicio do relatório dentro do arquivo _f1_p5_p10_map_r_prec_no_stemmer.csv_

## Métricas com stemmer
1. Métricas do conjunto de consultas
    - Mean Average Precision: 0.2506
    - Mean Reciprocal Rank: 0.8205
    - Average Discounted Gain: 0.1862
    - Normalized Discounted Gain: 0.4639
    - O gráfico de 11 pontos de precisão e recall médio para o conjunto de consultas pode ser encontrado na pasta mencionada anteriormente com o nome de _avg_11points_stemmer.jpg_ e _11points_stemmer.csv_

2. Métricas de consulta individuais
    -   O arquivo para análise de F1, P@5, P@10, 
    MAP para cada consulta e R-precision podem ser encontrados no diretório mencionado no inicio do relatório dentro do arquivo _f1_p5_p10_map_r_prec_stemmer.csv_

## Conclusões
Ao analisar o resultado das métricas de avaliação dos arquivos recuperados pelo modelo vetorial podemos observar que a técnica de stemming obteve os melhores resultados. Dado esse fato é visível que esta técnica se faz necessária em modelos de recuperação de informação.