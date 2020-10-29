# Exercício Modelo Vetorial

## Estrutura do projeto
1. A pasta _SRC_ contém todos os arquivos de execução do projeto, já _RESULT_ contém todos os arquivos gerados pelas execuções. Além disso em _EVALUATION_ há os arquivos gerados pela avaliação dos resultados do modelo vetorial.
2. Dentro da pasta _SRC_ temos os seguintes arquivos :
    - _generate_configs.py_ : Este script é utilizado para gerar os arquivos de configuração que serão utilizados em cada um dos passos do trabalho.
    
    - _queries_parser.py_ : Já este script é o responsável por processar os arquivos de query e construir os arquivos processados _expected.csv_ e _query.csv_.
    
    - _make_inverse_index.py_ : Neste programa é onde ocorre a geração da lista invertida onde esta é gerada a partir de todos os documentos xml providos para este trabalho, aqui geramos dois arquivos _inverted_index.csv_ e _doc_index.csv_ : Nós usamos este segundo arquivo para facilitar as contas na parte da geração do modelo e do cálculo de similaridade visto que estamos armazenando uma lista de adjacências e não uma matriz.
    
    - _create_model.py_ : Gera o modelo a ser utilizado para calcular a similaridade com a consulta, nesse caso cria o modelo vetorial.
    
    - _search_engine.py_ : Cria o arquivo de resultados baseado nos documentos com menor distância de cosseno entre eles e cada consulta utilizando o modelo vetorial. Limita a quantidade de documentos a maior quantidade de documentos esperados de uma consulta. Gera o arquivo _results_stemmer.json_ caso estejamos utilizando um stemmer ou gera _results_no_stemmer.json_ caso contrário.
    
    - _evaluate_results.py_ : Utiliza o arquivo de resultado gerado no script _search_engine.py_ para computar as seguintes métricas: F1, P@5, P@10, MRR, Curva de 11 pontos de precisão e recall, Avg DCG, NDCG, Histograma de R-precision e MAP. Os arquivos gerados podem ser encontrados no diretório _EVALUATION_.
    
    - _run.py_ : Executa todos os scripts previamente citados, para executar os scripts com o stemmer to Porter basta executá-lo com a flag _"-s yes"_.

3. Dentro da pasta _RESULT_ temos os seguintes arquivos :
    -  _query.csv_ : Consultas a serem realizadas.
    Linha ex: ```{"query_number": query_number ; "text": ['word_1','word_2', ... , 'word_n']; "max_expected_size":[max_value, max_value, ... , max_value]} ```

    -  _expected.csv_ : Score dos documentos esperados para a consulta especifica. Linha ex: ```{"query_numbers": query_number; "doc_number_list":['doc_1','doc_2', ... ,'doc_n']; "doc_score_list":['doc_1_score','doc_2_score', ... ,'doc_n_score']} ```

    -  _inverted_index.csv_ : Palavra por frequência de palavra em cada documento que ela aparece. Linha ex: ``` {"Word": word; "Adjacency Doc Frequency List": {'doc_1_word_apperas': freq_word_doc_1 ,'doc_2_word_apperas': freq_word_doc_2, ... , 'doc_n_word_apperas':freq_word_doc_n}}```

    -  _doc_index.csv_ : Documento por frequência de palavras no documento. Linha ex: ``` {"Doc": doc_number; "Adjacency Word Frequency List": {'Word_1_on_doc': freq_word_1 ,'Word_2_on_doc': freq_word_2, ... , 'Word_n_on_doc':freq_word_n}} ```

    -  _vector_space_model.json_ : Modelo vetorial contendo 3 informações para a realização das consultas de maneira prática. Linha ex: ```{
        'Inverse Document Frequency':
        {'WORD_1':idf_value, 'WORD_2': idf_value , ... }, 'Tf-Idf': {    
        'WORD_1':
        {'doc_1_word_1_appears': tf_idf_value, ...} ,
        'WORD_2':
        {'doc_1_word_2_appears': tf_idf_value, ... }, ... },'Doc Tf-Idf': 
        {'doc_1': {
            'word_1_on_doc_1': tf_idf_value, 'word_2_on_doc_1': tf_idf_value, ... }, 
        {'doc_2: {
            'word_1_on_doc_1': tf_idf_value, 'word_2_on_doc_1': tf_idf_value, ...}, ... 
        }
        }} ```

    -  _result__*_.json_ : Resultados com os documentos recuperados sobre cada consulta, limitado ao número maximo de documentos esperados por uma consulta. Linha ex: ```{"Query_number": query_number; "results": [ [position_1, doc_number, distance], ... , [position_n, doc_number, distance] ]}```

    -  _*.CFG_ : Arquivos de configuração

## Execução do projeto

Para executar o projeto, primeiramente é necessário instalar os pacotes que podem ser encontrados em _requirements.txt_. Após isso pode-se executar o script __run.py__ que ele executará todos os scripts em ordem que foram apresentados aqui. Para utilizar o stemmer do Porter pode-se utilizar _python run.py -s yes_ ou _python run.py --stemming yes_. **OBS: execute o código na pasta _SRC_**.