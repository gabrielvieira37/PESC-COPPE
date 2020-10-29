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
    LOGGER.info('Start making inverse index and document index')
    config = configparser.ConfigParser()
    parent_path = Path().resolve().parent
    config_path = str(parent_path) + '/RESULT/GLI.CFG'
    LOGGER.info(f"Read config path on : {config_path}")
    config.read(config_path)

    LOGGER.info(
        f"XML doc file paths: {config['GLI'].get('LEIA')}")
    LOGGER.info(
        f"Inverse index file and document index path: {config['GLI'].get('ESCREVA')}")

    files_list_str = config['GLI'].get('LEIA')
    # Load list from string
    files_list = ast.literal_eval(files_list_str)

    root = str(parent_path.parent) + '/'

    # stemmer placeholder
    stemmer = stemmer_obj

    docs_dict = {}
    # Create a set with stopwords with capital letters
    stop_words = [stop_word.upper()
                    for stop_word in stopwords.words('english')]
    words = set()
    # Get only words with length greater than 2
    # and without any number
    tokenizer = RegexpTokenizer(r'([A-Za-z]{3,})')
    for file_path in files_list:
        full_path = root + file_path
        LOGGER.info(f"Loading file: {full_path}")
        # load xml on a ElementTree
        tree = etree.parse(full_path)
        LOGGER.info("Loading all RECORD tags in this file")
        # from anywhere on the document find all tags named RECORD
        records = tree.findall('//RECORD')
        LOGGER.info(f"Founded {len(records)} RECORD tags in this file")
        LOGGER.info("Extracting ABSTRACT or EXTRACT from each tag, removing stopwords and counting word frequency.")
        for record in records:
            # For each record find recordnum
            # and check if there is abstract
            # if not check extract
            record_num = record.find('RECORDNUM')
            record_num = int(record_num.text)
            abstract = record.find('ABSTRACT')
            if abstract is None:
                extract = record.find('EXTRACT')
                if extract is None:
                    continue
                text = extract.text
                word_list = tokenizer.tokenize(text.upper())
                if stemmer:
                    word_list = [
                        stemmer.stem(word) for word in word_list if word not in stop_words]
                else:
                    word_list = [
                        word for word in word_list if word not in stop_words]
                word_frequency = Counter(word_list)
                docs_dict[record_num] = dict(word_frequency)

                # Create a list only with unique words
                words.update(word_list)
                continue
            text = abstract.text
            word_list = tokenizer.tokenize(text.upper())
            if stemmer:
                word_list = [
                    stemmer.stem(word) for word in word_list if word not in stop_words]
            else:
                word_list = [
                    word for word in word_list if word not in stop_words]
            word_frequency = Counter(word_list)
            docs_dict[record_num] = dict(word_frequency)

            # Create a list only with unique words
            words.update(word_list)

    # Make words be a list
    words = list(words)
    LOGGER.info("Finished parsing all documents")
    LOGGER.info(f"Founded {len(docs_dict)} documents and {len(words)} terms")
    LOGGER.info(f"Start creating inverted index")
    # Create inverted index
    inverted_index_dict = {}
    for word in words:
        doc_freq = {}
        for doc_num in docs_dict:
            freq = docs_dict[doc_num].get(word, 0)
            if freq > 0:
                doc_freq[doc_num] = freq
        inverted_index_dict[word] = doc_freq

    inverted_index = pd.DataFrame(inverted_index_dict.items(), columns=[
                                'Word', 'Adjacency Doc Frequency List'])
    LOGGER.info(f"Finished creating inverted index")


    output_files = ast.literal_eval(config['GLI'].get('ESCREVA'))
    inverted_csv_file_name = output_files[0]
    docs_csv_file_name = output_files[1]

    inverted_csv_file_name = str(parent_path) + inverted_csv_file_name
    docs_csv_file_name = str(parent_path) + docs_csv_file_name

    LOGGER.info(f'Writing inverted index on: {inverted_csv_file_name}')
    # Save inverted_index as csv
    inverted_index.to_csv(inverted_csv_file_name, sep=';', index=False)

    doc_index = pd.DataFrame(docs_dict.items(), columns=[
                            'Doc', 'Adjacency Word Frequency List'])

    LOGGER.info(f'Writing document index on : {docs_csv_file_name}')
    # Save doc_index as csv
    doc_index.to_csv(docs_csv_file_name, sep=';', index=False)
    LOGGER.info('Finished writting')
    LOGGER.info("Finishing program")


if __name__ == "__main__":
    main()
