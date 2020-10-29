import generate_configs
import queries_parser
import make_inverse_index
import create_model
import search_engine
import evaluate_results
import argparse
import logging
import nltk

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - { '
    '%(name)s - %(funcName)s(%(lineno)d) }: %(message)s')

def main(stemmer_obj):
    LOGGER.info("Starting run script")
    generate_configs.main(stemmer_obj)
    queries_parser.main(stemmer_obj)
    make_inverse_index.main(stemmer_obj)
    create_model.main()
    search_engine.main()
    evaluate_results.main(stemmer_obj)
    LOGGER.info("Finishing run script")

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-s", "--stemming",
                        help="Write 'yes' to activate stemming", default='no')

    ARGS = PARSER.parse_args()

    STEMER = ARGS.stemming

    if STEMER == "yes":
        LOGGER.info("Stemmer Activated")
        STEMER_OBJECT = nltk.stem.PorterStemmer()
    else:
        LOGGER.info("No stemmer being used")
        STEMER_OBJECT = False

    main(
        STEMER_OBJECT,
    )
