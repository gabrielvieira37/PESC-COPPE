import ast
import configparser
import json
import logging
import numpy as np
import pandas as pd
from collections import Counter
from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pathlib import Path


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - { '
    '%(name)s - %(funcName)s(%(lineno)d) }: %(message)s')

def main(stemmer_obj):
    LOGGER.info("Start parsing queries")
    config = configparser.ConfigParser()
    parent_path = Path().resolve().parent
    config_path = str(parent_path) + '/RESULT/PC.CFG'
    
    LOGGER.info(f"Read config path on : {config_path}")

    config.read(config_path)

    LOGGER.info(f"XML query file path: {config['PC'].get('LEIA')}")
    LOGGER.info(f"Parsed query file path: {config['PC'].get('CONSULTAS')}")
    LOGGER.info(f"Expected query file path:  {config['PC'].get('ESPERADOS')}")

    file_path = config['PC'].get('LEIA')
    root = parent_path.parent

    full_path = str(root) + '/' +file_path
    LOGGER.info(f'Full XML path: {root}')

    LOGGER.info(f'Start looking for QUERY element')
    # load xml on a ElementTree
    tree = etree.parse(full_path)
    # from anywhere on the document find all tags named QUERY
    queries = tree.findall('//QUERY')
    LOGGER.info(f'Found {len(queries)} queries elements')

    # stemmer placeholder
    stemmer = stemmer_obj

    # Create a set with stopwords with capital letters
    stop_words = [stop_word.upper() for stop_word in stopwords.words('english')]
    # Get only words with length greater than 2
    # and without any number
    tokenizer = RegexpTokenizer(r'([A-Za-z]{3,})')

    query_dict = {'query_number': [], 'text': []}
    expected_dict = {}

    LOGGER.info(f'Start parsing QUERY Elements')
    for query in queries:
        query_number = int(query.find('QueryNumber').text)
        text = query.find('QueryText').text
        word_list = tokenizer.tokenize(text.upper())
        if stemmer:
            word_list = [stemmer.stem(word) for word in word_list if word not in stop_words]
        else:
            word_list = [word for word in word_list if word not in stop_words]
        query_dict['query_number'].append(query_number)
        query_dict['text'].append(word_list)

        items = query.find('Records').findall('Item')

        expected_dict[query_number] = {
            'docs_result_list': [], 'score_result_list': []}
        # Giving less importance to author colleagues
        for item in items:
            # Score string
            score = item.values()[0]
            # Author weight
            auth = int(score[0]) * 2
            # Author colleagues weight
            auth_col = int(score[1])
            # Author Posdoc colleagues weight
            auth_pos = int(score[2]) * 2
            # Another Authors
            auth_ano = int(score[3]) * 2
            final_score = (auth + auth_col + auth_pos + auth_ano)/4
            # Document number
            doc_num = int(item.text)

            expected_dict[query_number]['docs_result_list'].append(doc_num)
            expected_dict[query_number]['score_result_list'].append(final_score)

        score_list = expected_dict[query_number]['score_result_list']
        # To make scalar operations on list
        score_array = np.array(score_list)
        # Reduce score into probability
        normalized_score_array = score_array/score_array.sum()
        expected_dict[query_number]['score_result_list'] = list(
            normalized_score_array)

    query_numbers = list(expected_dict.keys())
    docs_info =  list(expected_dict.values())
    doc_number_size_list = []
    expected_dict_for_df = {'query_number': [], 'doc_number_list':[],'doc_score_list':[]}
    for query_number,doc_info in zip(query_numbers, docs_info):
        expected_dict_for_df['query_number'].append(query_number)
        expected_dict_for_df['doc_number_list'].append(doc_info['docs_result_list'])
        expected_dict_for_df['doc_score_list'].append(doc_info['score_result_list'])
        doc_number_size_list.append(len(doc_info['docs_result_list']))
    
    max_expected_size = max(doc_number_size_list)
    query_dict['max_expected_size'] = [
        max_expected_size]*len(query_dict['query_number'])

    expected_df = pd.DataFrame(expected_dict_for_df)
    query_df = pd.DataFrame(query_dict)
    LOGGER.info(f'Finished parsing QUERY Elements')

    query_file = config['PC'].get('CONSULTAS')
    expected_file = config['PC'].get('ESPERADOS')

    query_file = str(parent_path) + query_file
    expected_file = str(parent_path) + expected_file


    query_df.to_csv(query_file, sep=';', index=False)
    LOGGER.info(f'Writing parsed queries on: {query_file}')
    expected_df.to_csv(expected_file, sep=';', index=False)
    LOGGER.info(f'Writing expected results on: {expected_file}')
    LOGGER.info('Saved parsed query and expected query on respective paths')
    LOGGER.info('Finished writting')
    LOGGER.info("Finishing program")

if __name__ == "__main__":
    main()
