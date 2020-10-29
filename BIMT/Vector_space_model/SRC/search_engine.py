import ast
import configparser
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - { '
    '%(name)s - %(funcName)s(%(lineno)d) }: %(message)s')

def main():
    LOGGER.info('Start search engine to generate query ranks')
    config = configparser.ConfigParser()
    parent_path = str(Path().resolve().parent)
    config_path = parent_path + '/RESULT/BUSCA.CFG'
    LOGGER.info(f"Read config path on : {config_path}")
    config.read(config_path)

    LOGGER.info(
        f"Model path: {config['BUSCA'].get('MODELO')}")
    LOGGER.info(
        f"Query path: {config['BUSCA'].get('CONSULTAS')}")
    LOGGER.info(
        f"Result path: {config['BUSCA'].get('RESULTADOS')}")


    model_file = config['BUSCA'].get('MODELO')
    model_file = parent_path + model_file
    query_file = config['BUSCA'].get('CONSULTAS')
    query_file = parent_path + query_file

    model = open(model_file)
    vectorial_model = json.load(model)
    query_df = pd.read_csv(query_file, sep=';')

    max_expected_size = query_df['max_expected_size'][0]
    LOGGER.info(
        "Start generating rankings for each query")
    # Store results for each query
    results = {}
    for idx, row in query_df.iterrows():
        query_number = row['query_number']
        query_words = ast.literal_eval(row['text'])
        query_length = 0
        doc_norm = {}
        rank_numerator = {}
        for word in query_words:
            # Get w_t_q
            word_query_weight = vectorial_model['Inverse Document Frequency'].get(
                word)
            
            if word_query_weight is None:
                continue
            # Query length = sum of all w_t_q**2
            query_length += pow(word_query_weight,2)
            tf_idf_word = vectorial_model['Tf-Idf'][word]
            for doc in tf_idf_word:
                # Get w_t_d
                word_document_weight = tf_idf_word[doc]
                if doc in rank_numerator:
                    
                    rank_numerator[doc] += word_document_weight * word_query_weight
                else:
                    # Create Doc length
                    rank_numerator[doc] = word_document_weight * word_query_weight

                if doc not in doc_norm:
                    # Get doc norms
                    doc_vector_length = sum(np.power(
                        list(vectorial_model['Doc Tf-Idf'][doc].values()), 2))
                    doc_norm[doc] = np.sqrt(doc_vector_length)
        # Get query norm
        query_norm = np.sqrt(query_length)

        rank = {}
        for doc, numerator in rank_numerator.items():
            # Rank each document that had at least
            # one word that query provided
            distance = numerator / (doc_norm[doc] * query_norm)
            rank[doc] = distance

        # Change rank to be [rank_position, doc_number, distance]
        rank_df = pd.Series(rank) \
            .sort_values(ascending=False) \
            .reset_index() \
            .reset_index() \
            .rename(columns={'level_0': 'rank', 'index': 'doc_number', 0: 'distance'}) \
            .iloc[:max_expected_size]
        rank_df['rank'] += 1
        rank_matrix = rank_df.to_numpy()
        # Store rank matrix
        results[query_number] = rank_matrix

    results_df = pd.DataFrame(results.items(), columns=['query_number', 'results'])
    LOGGER.info("Finished generating rankings")

    results_file = config['BUSCA'].get('RESULTADOS')
    results_file_path = parent_path + results_file
    LOGGER.info(f"Writing results file on: {results_file_path}")
    results_json = results_df.to_json()
    with open(results_file_path, "w") as results_file:
        json.dump(results_json, results_file)
    # results_df.to_csv(results_file, sep=';', index=False)
    LOGGER.info('Finished writting')
    LOGGER.info('Finishing program')

if __name__ == "__main__":
    main()
