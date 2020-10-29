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
    LOGGER.info("Starting model creation based on indexer")
    config = configparser.ConfigParser()
    parent_path = str(Path().resolve().parent)
    config_path = parent_path + '/RESULT/INDEX.CFG'
    LOGGER.info(f"Read config path on : {config_path}")
    config.read(config_path)

    LOGGER.info(
        f"Inverse index file and document index path: {config['INDEX'].get('LEIA')}")
    LOGGER.info(
        f"Model path: {config['INDEX'].get('ESCREVA')}")

    files_list_str = config['INDEX'].get('LEIA')
    # Load list from string
    files_list = ast.literal_eval(files_list_str)

    LOGGER.info(
        f"Loading inverse index: {parent_path+files_list[0]}")
    inverted_index = pd.read_csv(parent_path+files_list[0], sep=';')
    LOGGER.info(
        f"Loading document index: {parent_path+files_list[1]}")
    doc_index = pd.read_csv(parent_path+files_list[1], sep=';')


    # Sort words from a to z
    inverted_index.sort_values(by='Word', inplace=True)
    inverted_index.reset_index(drop=True, inplace=True)

    # Make a copy so we don't change dataframe
    tf_idf_df = inverted_index.copy()
    # Use word as a index
    tf_idf_df = tf_idf_df.set_index('Word')

    total_number_docs = doc_index['Doc'].count()
    word_idf_dict = {}
    word_tf_idf_dict = {}

    LOGGER.info("Start creating TF-IDF representation")
    # Doc Adjacency TF-IDF
    for word, row in tf_idf_df.iterrows():
        # Get frequency on docs that it appear
        word_frequencies_on_docs = ast.literal_eval(
            row['Adjacency Doc Frequency List'])
        # Get number of docs word apperas
        number_docs_word_appears = len(word_frequencies_on_docs)
        # Calculate inverse document frequency
        idf = np.log2(total_number_docs/number_docs_word_appears)
        word_idf_dict[word] = idf
        tf_idf_dict = {}
        for doc_number, frequency in word_frequencies_on_docs.items():
            # using third recommended tf-idf weighting scheme
            # implies using log normalization on term frequency
            normalized_term_frequency = 1 + np.log2(frequency)
            tf_idf_dict[doc_number] = normalized_term_frequency * idf

        # Put tf_idf for each word
        word_tf_idf_dict[word] = tf_idf_dict

    # Create series object to insert on dataframe
    idf_series = pd.Series(word_idf_dict, index=list(word_idf_dict.keys()))
    tf_idf_series = pd.Series(
        word_tf_idf_dict, index=list(word_tf_idf_dict.keys()))

    tf_idf_df['Inverse Document Frequency'] = idf_series
    tf_idf_df['Tf-Idf'] = tf_idf_series
    tf_idf_df.drop(columns=['Adjacency Doc Frequency List'], inplace=True)

    LOGGER.info("Start creating Doc TF-IDF representation")
    doc_tf_idf = doc_index.copy()
    doc_tf_idf.set_index('Doc', inplace=True)

    tf_idf_dict = {}
    # Word Adjacency TF-IDF
    for doc_number, row in doc_tf_idf.iterrows():
        word_frequencies = ast.literal_eval(row['Adjacency Word Frequency List'])
        tf_idf_word_dict = {}
        for word in word_frequencies:
            # Find already calculated Tf-IDF
            tf_idf_value = tf_idf_df['Tf-Idf'][word][doc_number]
            tf_idf_word_dict[word] = tf_idf_value
        tf_idf_dict[doc_number] = tf_idf_word_dict

    tf_idf_series = pd.Series(tf_idf_dict, index=list(tf_idf_dict.keys()))
    doc_tf_idf['Doc Tf-Idf'] = tf_idf_series

    doc_tf_idf.drop(columns=['Adjacency Word Frequency List'], inplace=True)

    LOGGER.info("Finished creating Doc TF-IDF representation")

    # Setting up a json to concatenate them
    doc_json_str = doc_tf_idf.to_json(orient="columns")
    vectorial_model_str = tf_idf_df.to_json(orient="columns")
    vectorial_model = json.loads(vectorial_model_str)
    doc_json = json.loads(doc_json_str)

    # Concatenating word adjacency with doc adjacency TF-IDF
    vectorial_model['Doc Tf-Idf'] = doc_json['Doc Tf-Idf']
    LOGGER.info("Finished creating TF-IDF representation")

    vectorial_model_file_path = config['INDEX'].get('ESCREVA')
    vectorial_model_file_path = parent_path + vectorial_model_file_path

    LOGGER.info(f'Writing vector space model on: {vectorial_model_file_path}')
    with open(vectorial_model_file_path, "w") as vectorial_model_file:
        json.dump(vectorial_model, vectorial_model_file)
    LOGGER.info('Finished writting')
    LOGGER.info('Finishing program')

if __name__ == "__main__":
    main()
