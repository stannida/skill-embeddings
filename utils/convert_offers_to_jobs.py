import pandas as pd
from collections import Counter
import os

def clean_title(title):
    return title.replace(' Großraum Berlin', '')

def assign_title_ids(df):
    title_index = {df['title'].unique()[x]:x for x in range(len(df['title'].unique()))}
    df['title_id'] = df['title'].apply(lambda x: title_index[x])

    return df

def map_ga_with_jobs(ga_data, df):
    
    ga_data['jobId'] = ga_data['jobId'].astype(int)

    company_data_merged = pd.merge(ga_data, df[['job_id','title_id']], how='left', left_on='jobId', right_on='job_id')
    company_data_merged = company_data_merged[~company_data_merged['title_id'].isna()]
    company_data_merged['title_id'] = company_data_merged['title_id'].astype(int)

    print(f'Number of unique titles found: {company_data_merged["title_id"].nunique()}')

    company_grouped = company_data_merged.groupby('clientId')

    similar_jobs = {}

    #one client would be interested in similar jobs, retrieving them in the format {job_A: [job_B, job_C, jb_B...], job_B:[]...}
    for clientId in company_data_merged['clientId'].unique():
        group = company_grouped.get_group(clientId)
        if len(set(group['title_id']))>1:
            for title_id in set(group['title_id']):
                sim_temp = list(group['title_id'].unique())
                sim_temp.remove(title_id)
                if title_id in similar_jobs.keys():
                    similar_jobs[title_id].extend(sim_temp)
                else:
                    similar_jobs[title_id] = sim_temp


    similar_jobs = {k: v for k, v in similar_jobs.items() if len(v) >= 2}

    #sort and append to the new dict
    similar_jobs_full = {}
    for job in similar_jobs.keys():
        count = Counter(similar_jobs[job])
        target_ids = list(count.keys())
        if len(target_ids)>1:
            similar_jobs_full[job] = target_ids

    #check that the values in eachlist are in the dict keys
    for job in similar_jobs_full.keys():
        remove_non_values = [_id for _id in similar_jobs_full[job] if _id in similar_jobs_full.keys()]
        similar_jobs_full[job] = remove_non_values

    #remove those that are less than 1
    similar_jobs_full = {k: v for k, v in similar_jobs_full.items() if len(v) >= 1}

    print(f'Number of query jobs found: {len(similar_jobs_full.keys())}')
    return similar_jobs_full, company_data_merged

def convert_and_save(unique_jobs, similar_jobs_full, name):
    temp = {'id':[],'query':[]}


    for job in similar_jobs_full.keys():
        if job in unique_jobs['title_id'].values:
            temp['id'].append(job)
            temp['query'].append(unique_jobs.loc[unique_jobs['title_id']==job, 'texts'].values[0])
        else:
            temp['id'].append(int(job))
            temp['query'].append('JOB NOT FOUND')
        for i in range(10):
            try:
                sim_job_id = similar_jobs_full[job][i]
            except:
                sim_job_id = None
            if 'sim'+str(i) in temp.keys():
                temp['sim'+str(i)].append(sim_job_id)
                try:
                    temp['title'+str(i)].append(unique_jobs.loc[unique_jobs['title_id']==int(sim_job_id), 'texts'].values[0])
                except:
                    temp['title'+str(i)].append('JOB NOT FOUND')
            else:
                temp['sim'+str(i)] = [sim_job_id]
                try:
                    temp['title'+str(i)] = [unique_jobs.loc[unique_jobs['title_id']==int(sim_job_id), 'texts'].values[0]]
                except:
                    temp['title'+str(i)] = ['JOB NOT FOUND']

    pd.DataFrame(temp).to_excel(os.path.join('resources/20210318', name))
