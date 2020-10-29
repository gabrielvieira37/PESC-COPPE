import configparser
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - { '
    '%(name)s - %(funcName)s(%(lineno)d) }: %(message)s')

def main(stemmer_obj):
    parent_path = str(Path().resolve().parent)

    LOGGER.info(
        f"Generate query parser config file on: {parent_path + '/RESULT/PC.CFG'}")
    config = configparser.ConfigParser()
    config['PC'] = {'LEIA': 'Homework_XML/data/cfquery.xml'}
    config['PC']['CONSULTAS'] = '/RESULT/query.csv'
    config['PC']['ESPERADOS'] = '/RESULT/expected.csv'
    with open(parent_path + '/RESULT/PC.CFG', 'w') as configfile:
        config.write(configfile)

    LOGGER.info(
        f"Generate GLI config file on: {parent_path + '/RESULT/GLI.CFG'}")
    config = configparser.ConfigParser()
    config['GLI'] = {'LEIA': ['Homework_XML/data/cf74.xml', 'Homework_XML/data/cf79.xml', 'Homework_XML/data/cf75.xml',
                            'Homework_XML/data/cf76.xml', 'Homework_XML/data/cf77.xml', 'Homework_XML/data/cf78.xml']}
    config['GLI']['ESCREVA'] = "['/RESULT/inverted_index.csv', '/RESULT/doc_index.csv']"
    with open(parent_path + '/RESULT/GLI.CFG', 'w') as configfile:
        config.write(configfile)

    LOGGER.info(
        f"Generate indexer config file on: {parent_path + '/RESULT/INDEX.CFG'}")
    config = configparser.ConfigParser()
    config['INDEX'] = {
        'LEIA': ['/RESULT/inverted_index.csv', '/RESULT/doc_index.csv']}
    config['INDEX']['ESCREVA'] = "/RESULT/vector_space_model.json"
    with open(parent_path + '/RESULT/INDEX.CFG', 'w') as configfile:
        config.write(configfile)

    LOGGER.info(
        f"Generate search engine config file on: {parent_path + '/RESULT/BUSCA.CFG'}")
    config = configparser.ConfigParser()
    config['BUSCA'] = {'MODELO': '/RESULT/vector_space_model.json'}
    config['BUSCA']['CONSULTAS'] = "/RESULT/query.csv"
    if stemmer_obj:
        config['BUSCA']['RESULTADOS'] = "/RESULT/results_stemmer.json"
    else:
        config['BUSCA']['RESULTADOS'] = "/RESULT/results_no_stemmer.json"
    with open(parent_path + '/RESULT/BUSCA.CFG', 'w') as configfile:
        config.write(configfile)

    LOGGER.info(
        f"Generate search engine config file on: {parent_path + '/RESULT/EVAL.CFG'}")
    config = configparser.ConfigParser()
    config['EVAL'] = {'ESPERADOS': '/RESULT/expected.csv'}
    config['EVAL']['ESCREVE'] = '/EVALUATION/'
    if stemmer_obj:
        config['EVAL']['RESULTADOS'] = "/RESULT/results_stemmer.json"
    else:
        config['EVAL']['RESULTADOS'] = "/RESULT/results_no_stemmer.json"
    with open(parent_path + '/RESULT/EVAL.CFG', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    main()
