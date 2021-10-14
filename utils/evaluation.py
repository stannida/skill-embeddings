import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score


def get_map(gold_standard_dict, predictions):
    multibinarizer = MultiLabelBinarizer()

    true = multibinarizer.fit(list(gold_standard_dict.values())).transform(list(gold_standard_dict.values()))
    pred = multibinarizer.transform(list(predictions.values()))

    average_precision = average_precision_score(true,pred,average='samples')
    average_recall = recall_score(true,pred,average='samples', zero_division=0)

    return average_precision, average_recall

def arg_sort_threshold(x, thresh, reverse, max_n=11, min_n=3):
    idx = np.arange(x.size)[x > thresh]
    if len(idx)<min_n:
        if reverse == False:
            return np.argsort(x)[::-1][:min_n]
        else:
            return np.argsort(x)[:min_n]
    elif len(idx)>max_n:
        if reverse == False:
            return np.argsort(x)[::-1][:max_n]
        else:
            return np.argsort(x)[:max_n]
    if reverse == False:
        return np.argsort(x[x > thresh])[::-1]
    else:
        return np.argsort(x[x > thresh])




def get_predicted_results(gold_standard_dict, dist_out, thresh, reverse=False, max_n=10, min_n=2):
    # getting the results of the distance matrix in the same form as gold standard
    predictions = {}
    for q in gold_standard_dict.keys():
        vector = np.copy(dist_out[q])
        #sort the distance vector for each query and retrieve the indexes
        true = arg_sort_threshold(vector, thresh, reverse)
        #select top n target indexes (here a threshold needs to be implemented)
        predictions[q] = true[1:]
    
    return predictions


def get_precision_recall(gold_standard_dict, predictions):
    multibinarizer = MultiLabelBinarizer()

    true = multibinarizer.fit(list(gold_standard_dict.values())).transform(list(gold_standard_dict.values()))
    pred = multibinarizer.transform(list(vdf_predictions.values()))

    average_precision_score(true,pred,average='samples')
    return avg_precision, avg_recall


def get_f_score(true, pred, beta):
    f_score = fbeta_score(true, pred, average='samples', beta=beta)
    return f_score

def retrieve_similar_jobs(dist_matrix, gold_standard_dict, thresh, min_n=3, max_n=11):
    predictions = {}
    for job_id in gold_standard_dict.keys():

        sorted_ids = dist_matrix[job_id].sort_values(ascending=False)
        idx = sorted_ids[sorted_ids>thresh].index.tolist()
        if job_id in idx: idx.remove(job_id)
        if len(idx)<min_n:
            idx = sorted_ids[:min_n+1].index.tolist()
            if job_id in idx: idx.remove(job_id)
            predictions[job_id] = idx
        elif len(idx)>max_n:
            predictions[job_id] = idx[:max_n]
        else:
            predictions[job_id] = idx
    return predictions