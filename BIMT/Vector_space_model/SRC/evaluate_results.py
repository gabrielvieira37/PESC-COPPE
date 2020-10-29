import ast
import configparser
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - { '
    '%(name)s - %(funcName)s(%(lineno)d) }: %(message)s')

def rank_and_expected_rank(row, results_df, query_number):
    """
    Return rank_df and expected_rank_df
    """
    doc_number_list = ast.literal_eval(row['doc_number_list'])
    score_list = ast.literal_eval(row['doc_score_list'])
    size = len(doc_number_list)
    rank = list(range(1, size+1))
    expected_rank = {'doc_number': doc_number_list, 'score': score_list}
    expected_rank_df = pd.DataFrame(expected_rank)
    expected_rank_df = expected_rank_df.sort_values(
        by='score', ascending=False).reset_index(drop=True)
    expected_rank_df['rank'] = rank

    results = results_df['results'][query_number]
    rank_df = pd.DataFrame(results, columns=['rank', 'doc_number', 'distance'])
    rank_df['doc_number'] = rank_df['doc_number'].astype(int)

    return rank_df, expected_rank_df


def recall_precision(rank_doc_set, expected_rank_doc_set):
    """
    calculate recall and precision given rank
    and expected rank
    """
    relevant_documents_retrieved = rank_doc_set.intersection(
        expected_rank_doc_set)
    relevant_doc_ret_size = len(relevant_documents_retrieved)
    retrieved_doc_size = len(rank_doc_set)
    relevant_doc_size = len(expected_rank_doc_set)

    # Precision
    precision = relevant_doc_ret_size/retrieved_doc_size

    # Recall
    recall = relevant_doc_ret_size/relevant_doc_size

    return precision, recall


def rel_pres_recall(rank_df, expected_rank_doc_set):
    """
    Return recall and precision for each relevant found
    on its rank position
    """
    relevant_precis_recall = {'index': [], 'precision': [], 'recall': []}
    for _, rank_row in rank_df.iterrows():
        if rank_row['doc_number'] in list(expected_rank_doc_set):
            mask = rank_df['doc_number'] == rank_row['doc_number']
            slice_index = rank_df[mask].index[0]
            slice_index += 1
            rank_doc_set_sliced = set(rank_df.iloc[:slice_index]['doc_number'])

            precision, recall = recall_precision(
                rank_doc_set_sliced, expected_rank_doc_set)

            relevant_precis_recall['index'].append(slice_index)
            relevant_precis_recall['precision'].append(precision)
            relevant_precis_recall['recall'].append(recall)
    return relevant_precis_recall


def recall_precision_11_points(relevant_precis_recall):
    """
    Create 11 points precision recall curve using
    recall and precision matri for each relevant 
    position
    """
    recall_axis = np.linspace(0.0, 1.0, num=11)
    rec_prec_11_points = np.zeros((11, 2))
    precision_max = 0
    for index, recall_point in enumerate(recall_axis):
        flag_inserted = 0
        for position, recall in enumerate(relevant_precis_recall['recall']):
            precision = relevant_precis_recall['precision'][position]
            if recall >= recall_point:
                # Retrieve only max precision point from any recall >= recall%
                precision_max = max(
                    relevant_precis_recall['precision'][position:])
                # index 0 = recall values
                rec_prec_11_points[index][0] = recall_point
                # index 1 = precision values
                rec_prec_11_points[index][1] = precision_max
                flag_inserted = 1
                break

        # if didnt insert anything copy last precision point
        if not flag_inserted:
            rec_prec_11_points[index][0] = recall_point
            rec_prec_11_points[index][1] = rec_prec_11_points[index-1][1]

    return rec_prec_11_points


def calculate_r_precision(rank_df, expected_rank_doc_set):
    """
    Calculate precision on rank using relevant size
    """
    relevant_size = len(expected_rank_doc_set)
    rank_doc_set_sliced = set(rank_df.iloc[:relevant_size]['doc_number'])
    r_precision, _ = recall_precision(
        rank_doc_set_sliced, expected_rank_doc_set)
    return r_precision


def calculate_reciprocal_rank(rank_df, expected_rank_df):
    """
    Calculate reciprocal rank = 1/first_rank_relevant
    """
    counter = 0
    for relevant_doc in expected_rank_df['doc_number']:
        if relevant_doc in list(rank_df['doc_number']):
            mask = rank_df['doc_number'] == relevant_doc
            rank = int(rank_df.loc[mask]['rank'])
            if counter == 0:
                counter = 1
                rank_min = rank
            if rank < rank_min:
                rank_min = rank
    reciprocal_rank = 1/rank_min
    return reciprocal_rank


def calculate_dcg(rank_df, expected_rank_df):
    """
    Calculate dcg using score from expected_rank_df
    """
    for index, rank_row in rank_df.iterrows():
        # Rank only to 10
        if index > 10:
            break
        rank = rank_row['rank']
        doc_number = rank_row['doc_number']
        score = 0
        if doc_number in list(expected_rank_df['doc_number']):
            mask = expected_rank_df['doc_number'] == doc_number
            score = float(expected_rank_df[mask]['score'])
        if rank > 1:
            discounted_cumulative_gain += score/np.log2(rank)
            continue
        discounted_cumulative_gain = score
    return discounted_cumulative_gain


def calculate_idcg(expected_rank_df):
    """
    Calculate idealized discounted 
    gain using expected_rank_df
    score
    """
    for index, expected_row in expected_rank_df.iterrows():
        # Rank only to 10
        if index > 10:
            break
        rank = expected_row['rank']
        if rank > 1:
            idealized_discounted_cumulative_gain += expected_row['score']/np.log2(
                rank)
            continue
        idealized_discounted_cumulative_gain = expected_row['score']
    return idealized_discounted_cumulative_gain


def main(stemmer_object):
    root = Path().resolve().parent

    LOGGER.info("Starting results evaluation")
    config = configparser.ConfigParser()
    parent_path = str(Path().resolve().parent)
    config_path = parent_path + '/RESULT/EVAL.CFG'
    LOGGER.info(f"Read config path on : {config_path}")
    config.read(config_path)

    LOGGER.info(
        f"Results from vector space model on: {config['EVAL'].get('RESULTADOS')}")
    LOGGER.info(
        f"Expected results file path: {config['EVAL'].get('ESPERADOS')}")
    LOGGER.info(
        f"Evaluation file path: {config['EVAL'].get('ESCREVE')}")

    results_file_path = config['EVAL'].get('RESULTADOS')
    expected_file_path = config['EVAL'].get('ESPERADOS')
    eval_files_path = config['EVAL'].get('ESCREVE')

    results_file = parent_path + results_file_path
    expected_file = parent_path + expected_file_path
    eval_files_path = parent_path + eval_files_path

    results = open(results_file)
    results_json_str = json.load(results)
    results_json = json.loads(results_json_str)
    results_df = pd.DataFrame(results_json)
    expected_df = pd.read_csv(expected_file, sep=';')

    results_df.set_index('query_number', inplace=True)

    if stemmer_object:
        stemmer_present = 'stemmer'
    else:
        stemmer_present = 'no_stemmer'

    query_eval = {}
    mean_average_precision_total = []
    r_precision_total = []
    reciprocal_rank_total = []
    dcg_total = []
    idcg_total = []
    rec_prec_11_points_total = []
    LOGGER.info(f"Calculating all metrics from {results_file_path}")
    for idx, row in tqdm(expected_df.iterrows()):
        evaluations = {}
        query_number = row['query_number']

        rank_df, expected_rank_df = rank_and_expected_rank(
            row, results_df, query_number)

        expected_rank_doc_set = set(expected_rank_df['doc_number'])
        rank_doc_set = set(rank_df['doc_number'])

        precision, recall = recall_precision(rank_doc_set, expected_rank_doc_set)

        # F1
        epsilon = 1e-10
        f1 = 2/((1/(recall+epsilon))+(1/(precision+epsilon)))
        evaluations['f1'] = f1

        top_5_doc_rank = set(rank_df['doc_number'][:5])
        precision_at_5 = len(top_5_doc_rank.intersection(expected_rank_doc_set))/5

        top_10_doc_rank = set(rank_df['doc_number'][:10])
        precision_at_10 = len(
            top_10_doc_rank.intersection(expected_rank_doc_set))/10

        evaluations['P@5'] = precision_at_5
        evaluations['P@10'] = precision_at_10

        relevant_precis_recall = rel_pres_recall(rank_df, expected_rank_doc_set)

        rec_prec_11_points = recall_precision_11_points(relevant_precis_recall)
        rec_prec_11_points = recall_precision_11_points(relevant_precis_recall)
        rec_prec_11_points_total.append(rec_prec_11_points)
        
        r_precision = calculate_r_precision(rank_df, expected_rank_doc_set)
        r_precision_total.append(r_precision)

        relevant_size = len(expected_rank_doc_set)
        mean_average_precision_query = sum(
            relevant_precis_recall['precision'])/relevant_size
        mean_average_precision_total.append(mean_average_precision_query)

        reciprocal_rank = calculate_reciprocal_rank(rank_df, expected_rank_df)
        reciprocal_rank_total.append(reciprocal_rank)

        discounted_cumulative_gain = calculate_dcg(rank_df, expected_rank_df)
        dcg_total.append(discounted_cumulative_gain)

        idealized_discounted_cumulative_gain = calculate_idcg(expected_rank_df)
        idcg_total.append(idealized_discounted_cumulative_gain)

        query_eval[query_number] = evaluations

    LOGGER.info("Finished calculating metrics")
    LOGGER.info("Saving metrics")
    mean_reciprocal_rank = sum(
        reciprocal_rank_total)/len(reciprocal_rank_total)
    mean_average_precision = sum(
        mean_average_precision_total)/len(mean_average_precision_total)
    avg_dcg_10 = sum(dcg_total)/len(dcg_total)

    avg_idcg_10 = sum(idcg_total)/len(idcg_total)
    normalized_dcg_10 = avg_dcg_10/avg_idcg_10

    query_eval_df = pd.DataFrame(query_eval.values(), index=query_eval.keys())
    query_eval_df['MAP'] = mean_average_precision_total
    query_eval_df['R_precision'] = r_precision_total
    query_eval_df.to_csv(
        f"{eval_files_path}/f1_p5_p10_map_r_prec_{stemmer_present}.csv")

    x = list(expected_df['query_number'])
    y = r_precision_total
    plt.figure(figsize=(12, 9))
    plt.title('R-Precision Histogram')
    plt.ylabel('R-Precision')
    plt.xlabel('Query number')
    plt.bar(x, y)
    plt.grid()
    plt.savefig(f'{eval_files_path}r_precision_histogram_{stemmer_present}.png')
    plt.close()

    avg_rec_prec_11_points = np.zeros_like(rec_prec_11_points_total[0])
    for rec_prec_11_points in rec_prec_11_points_total:
        avg_rec_prec_11_points += rec_prec_11_points
    avg_rec_prec_11_points = avg_rec_prec_11_points/len(rec_prec_11_points_total)

    plt.figure(figsize=(12, 9))
    plt.title('11-points AVG Recall Precision')
    plt.ylabel('Precision')
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlabel('Recall')
    plt.step(avg_rec_prec_11_points[:, 0],
             avg_rec_prec_11_points[:, 1], 'o-', c='b', alpha=0.7)
    plt.grid()
    plt.savefig(
        f'{eval_files_path}avg_11points_{stemmer_present}.png')
    plt.close()

    df_11_points = pd.DataFrame(rec_prec_11_points).rename(
            columns={0: 'Recall', 1: 'Precision'})
    df_11_points.to_csv(
        f"{eval_files_path}11points_{stemmer_present}.csv", index=False)

    LOGGER.info(
        f"MRR: {mean_reciprocal_rank:.4f}, AVG_DCG_10: {avg_dcg_10:.4f}, "+
        f"NDCG_10: {normalized_dcg_10:.4f}, MAP: {mean_average_precision:.4f}")
    LOGGER.info("Finished saving all metrics")
    LOGGER.info("Finishing program")

if __name__ == "__main__":
    main()
