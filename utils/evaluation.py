import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score
sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/include')
sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/topic_ontology')
sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/')
import mixedPipeline as mp
from mixedPipelineUtils.performance_measures import measureperformance

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

def get_map_score(dfJobSim, gold_standard_dict):

    map_values = []
    re_values = []
    map_thresh = []
    for thresh in np.arange(0, 1.05, 0.05):
        predictions = retrieve_similar_jobs(dfJobSim, gold_standard_dict, thresh, min_n=2, max_n=10)
        map_precision, recall = get_map(gold_standard_dict, predictions)
        map_thresh.append(thresh)
        map_values.append(map_precision)
        re_values.append(recall)

    max_map = max(map_values)

    return max_map, map_thresh[map_values.index(max_map)], max(re_values), map_thresh[re_values.index(max(re_values))]

def get_performance_scores(word_vectors, companyDataset, args, gold_standard_df, gold_standard_dict, skills_annotated_sample):

    attract_repel_skills = pd.DataFrame([word_vectors]).transpose().reset_index()
    attract_repel_skills.rename(columns={'index':'id', 0:'vector'}, inplace=True)
    attract_repel_skills.set_index('id', inplace=True)

    topicsSimilarities = mp.computeTopicSimilarities(attract_repel_skills)


    dfJobSim = mp.compareJobs(companyDataset,topicsSimilarities,eval('mp.simSkillSet'),**args)

    roc_score = measureperformance(dfJobSim, gold_standard_df)
    map_score, map_thresh, max_recall, recall_thresh = get_map_score(dfJobSim, gold_standard_dict)

    skills_annotated_sample['orig_skill_we'] = skills_annotated_sample['orig_skill'].apply(lambda x: word_vectors[x])
    skills_annotated_sample['skill1_we'] = skills_annotated_sample['skill_1'].apply(lambda x: word_vectors[x])
    skills_annotated_sample['skill2_we'] = skills_annotated_sample['skill_2'].apply(lambda x: word_vectors[x])
    skills_annotated_sample['cosine1'] = skills_annotated_sample.apply(lambda x: calc_cosine(x['orig_skill_we'], x['skill1_we']), axis=1)
    skills_annotated_sample['cosine2'] = skills_annotated_sample.apply(lambda x: calc_cosine(x['orig_skill_we'], x['skill2_we']), axis=1)
    skills_annotated_sample['similar_skill'] = skills_annotated_sample.apply(lambda x: x['skill_1'] if x['cosine1']>x['cosine2'] else x['skill_2'], axis=1)
    jaccard = jaccard_score(skills_annotated_sample['Similar annotation'].to_numpy(), skills_annotated_sample['similar_skill'].to_numpy(), average='macro')

    return roc_score['roc_auc'], map_score, map_thresh, max_recall, recall_thresh, jaccard
