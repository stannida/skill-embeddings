import pandas as pd
import itertools
import random

from helper_functions import calc_cosine
from tc_data_libs.utils.utils import initNeo4jClient, runPy2neoQuery

neo4j = initNeo4jClient({'host_neo4j': '51.89.20.204',
                         'port': 7474, 'user_neo4j': 'neo4j', 'password_neo4j': 'Semantic2020'})

query_all_ids = '''
    MATCH (n:MSkill) 
    WHERE n.type='SKILL' 
    MATCH (n)-[we:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN id(n) as ids, we.vec as embeddings
'''

df_ids,r = runPy2neoQuery(neo4j,query_all_ids)
print(len(df_ids))

random_state = 42
N = 200

node_ids = df_ids.sample(n= N, random_state = random_state)

query_siblings = '''
    UNWIND $batch AS row
    MATCH (o_skill:MSkill)-[:BROADER]->(p:MSkill)<-[:BROADER]-(sib_skill:MSkill)
    WHERE id(o_skill) = row['ids']
    MATCH (sib_skill)-[we:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN o_skill.name as o_name, id(o_skill) as o_id, sib_skill.name as names, id(sib_skill) as ids, we.vec as embeddings

'''

df_siblings,r = runPy2neoQuery(neo4j,query_siblings, batch=node_ids.to_dict(orient='records'))

df_siblings_sample = df_siblings.groupby(['o_id','o_name'], group_keys=False).apply(pd.DataFrame.sample, n=1, random_state=random_state)
print(len(df_siblings_sample))

query_not_siblings = '''
    UNWIND $batch AS row
    MATCH (o_skill:MSkill)
    WHERE id(o_skill) = row['ids']
    MATCH (skill:MSkill)-[we:CODED_IN]->(:SemanticModel {id:'GoogleWE'})
    WHERE NOT (o_skill)-[:BROADER]->(:MSkill)<-[:BROADER]-(skill)
    RETURN o_skill.name as o_name, id(o_skill) as o_id, skill.name as names, id(skill) as ids, we.vec as embeddings, rand() as r
    ORDER BY r LIMIT 30000

'''

df_not_siblings,r = runPy2neoQuery(neo4j,query_not_siblings, batch=node_ids.to_dict(orient='records'))

num_random_nodes = 100

df_nodes_sample = df_not_siblings.groupby(['o_id', 'o_name'], group_keys=False).apply(pd.DataFrame.sample, n=num_random_nodes, random_state=random_state)
print(len(df_nodes_sample))

merged_df = pd.merge(df_nodes_sample, node_ids, how='left', left_on='o_id', right_on='ids', suffixes=('_sample', '_orig'))
merged_df = merged_df[~merged_df['embeddings_sample'].isna()]
merged_df['cosine'] = merged_df.apply(lambda x: calc_cosine(x['embeddings_sample'], x['embeddings_orig']), axis=1)

closest = merged_df.loc[merged_df.groupby(['o_name','o_id'])["cosine"].idxmax()]  
print(len(closest))

pair_df = pd.merge(df_siblings_sample[['o_name','o_id','names']], closest[['o_name','o_id','names']], how='left', on=['o_id', 'o_name'], suffixes=('_sibling', '_nonsiblings'))
pair_df.to_excel('../resources/set_random_skills_truth.xlsx')

similar_dict = {'orig_skill':[], 'skill_1':[], 'skill_2':[]}
for i, row in pair_df.iterrows():
    r = [row['names_sibling'],row['names_nonsiblings']]
    random.shuffle(r)
    similar_dict['orig_skill'].append(row['o_name'])
    similar_dict['skill_1'].append(r[0])
    similar_dict['skill_2'].append(r[1])

pd.DataFrame(similar_dict).to_excel('../resources/set_random_skills.xlsx')
print('data is stored')






# to-do: 
# get closest non-sibling by cosine similarity
# save as a dict {o_name: (sibling name, non-sibling name)}
# write to excel file: o_name, sibling/non-sibling name, shuffle randomly


