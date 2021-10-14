import pandas as pd
import numpy as np
from ast import literal_eval
import pickle5 as pickle
import sys
import os
import json
from sklearn.metrics import jaccard_score

sys.path.insert(1, '/Users/astankevich/Desktop/thesis/tc_skills')
from utils.helper_functions import calc_cosine
from utils.evaluation import *

sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/include')
sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/topic_ontology')
sys.path.insert(1, '/Users/astankevich/tc_libraries/motherdb/')
import mixedPipeline as mp
from mixedPipelineUtils.performance_measures import measureperformance

def read_vectors(vecfile):

    attract_repel_skills = pd.read_csv(vecfile, sep=" ", header=None)
    attract_repel_skills['vector'] = attract_repel_skills[attract_repel_skills.columns[1:]].values.tolist()
    attract_repel_skills.rename(columns={0:'id'}, inplace=True)
    attract_repel_skills = attract_repel_skills[['id', 'vector']].set_index('id')

    attract_margin = vecfile.split('-')[2]
    batch_size = vecfile.split('-')[5]
    l2 = vecfile.split('-')[7][:-4]
    attract_repel_dict = {"attract_margin":attract_margin,
                            "batch_size":batch_size,
                            "l2":l2,
                            "df":attract_repel_skills}

    return attract_repel_dict

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


def get_metrics(attract_repel_skills, companyDataset, gold_standard_df, gold_standard_dict, skills_annotated_sample, roc_auc_args):

    companyDataset['skills'] = companyDataset['skills'].apply(lambda x: [word for word in x if word in attract_repel_skills.index.tolist()])

    topicsSimilarities = mp.computeTopicSimilarities(attract_repel_skills)
    
    dfJobSim = mp.compareJobs(companyDataset,topicsSimilarities,eval('mp.simSkillSet'),**roc_auc_args)

    roc_score = measureperformance(dfJobSim, gold_standard_df)
    map_score, map_thresh, max_recall, recall_thresh = get_map_score(dfJobSim, gold_standard_dict)

    skills_annotated_sample['orig_skill_we'] = skills_annotated_sample['orig_skill'].apply(lambda x: attract_repel_skills.loc[x])
    skills_annotated_sample['skill1_we'] = skills_annotated_sample['skill_1'].apply(lambda x: attract_repel_skills.loc[x])
    skills_annotated_sample['skill2_we'] = skills_annotated_sample['skill_2'].apply(lambda x: attract_repel_skills.loc[x])
    skills_annotated_sample['cosine1'] = skills_annotated_sample.apply(lambda x: calc_cosine(x['orig_skill_we'], x['skill1_we']), axis=1)
    skills_annotated_sample['cosine2'] = skills_annotated_sample.apply(lambda x: calc_cosine(x['orig_skill_we'], x['skill2_we']), axis=1)
    skills_annotated_sample['similar_skill'] = skills_annotated_sample.apply(lambda x: x['skill_1'] if x['cosine1']>x['cosine2'] else x['skill_2'], axis=1)
    agreement = len(skills_annotated_sample[skills_annotated_sample['similar_skill']==skills_annotated_sample['Similar annotation']])/len(skills_annotated_sample)
    jaccard = jaccard_score(skills_annotated_sample['Similar annotation'].to_numpy(), skills_annotated_sample['similar_skill'].to_numpy(), average='macro')


    return roc_score, map_score, map_thresh, max_recall, recall_thresh, jaccard

if __name__ == "__main__":
    gold_standard = pd.read_excel('resources/20210318/sim_jobs_with_skills.xlsx', dtype=str)
    gold_standard_jobs = gold_standard['id'].tolist()

    gold_standard_dict = {}
    for i, row in gold_standard.iterrows():
        query_id = row['id']
        similar_jobs = row.filter(regex=("sim.*")).dropna().tolist()
        gold_standard_jobs.extend(similar_jobs)
        gold_standard_dict[query_id] = similar_jobs

    gold_standard_df = pd.DataFrame([gold_standard_dict]).transpose().reset_index()
    gold_standard_df.rename(columns={0:'recs', 'index':'job'}, inplace=True)
    gold_standard_df.set_index('job', inplace=True)
    gold_standard_jobs = list(set(gold_standard_jobs))

    companyDataset = pd.read_excel('resources/company_profiles_with_skills.xlsx')
    companyDataset['po_id'] = companyDataset['po_id'].astype(str)
    companyDataset = companyDataset[companyDataset['po_id'].isin(gold_standard_jobs)]
    companyDataset.set_index('po_id', inplace=True)
    companyDataset['skills'] = companyDataset['skills'].apply(literal_eval)
    companyDataset['skills'] = companyDataset['skills'].apply(lambda x: [word.replace(' ', '_') for word in x])


    args={"threshold_scores":[{'threshold': 0.8, 'score': 1.0},
    {'threshold': 0.75, 'score': 0.75},
    {'threshold': 0.65, 'score': 0.5},
    {'threshold': 0.6, 'score': 0.25}]}

    annotations_alex = pd.read_excel('resources/set_random_skills.xlsx')
    annotations_ivan = pd.read_excel('resources/set_random_skills Ivan.xlsx')
    truth = pd.read_excel('resources/set_random_skills_truth.xlsx')
    skills_annotations = annotations_alex[annotations_alex['Similar annotation']==annotations_ivan['Similar annotation']]

    print("started evaluating the models")
    directory = "attract-repel/results/grid_search"


    attract_repel_results = pd.read_json(directory+'/results.json', orient='records')
    attract_repel_results_we = attract_repel_results[attract_repel_results['random']=='False']
    attract_repel_results['avg_score'] = (attract_repel_results['roc_score']+attract_repel_results['map_score']+attract_repel_results['agreement'])/3
    attract_repel_results_we = attract_repel_results[attract_repel_results['random']=='False']
    print(attract_repel_results_we.iloc[attract_repel_results_we['avg_score'].idxmax()])




