import pandas as pd
import itertools
import json

from tc_data_libs.utils.utils import initNeo4jClient, runPy2neoQuery
from helper_functions import calc_cosine


with open('attract-repel/config/neo4j_config.json') as f:
  neo4j_config = json.load(f)

neo4j = initNeo4jClient(neo4j_config)



def append_category_nodes(df, sub_category_nodes):
    for i, row in sub_category_nodes.iterrows():
        if len(row['c.name'])>1:
            for pair in itertools.combinations(row['c.name'],2):
                vec_c = df.loc[df['c.name']==pair[0], 'we_c.vec'].values[0]
                vec_p = df.loc[df['c.name']==pair[1], 'we_c.vec'].values[0]

                df = df.append({'c.name':pair[0], 'p.name':pair[1], 'we_c.vec':vec_c, 'we_p.vec':vec_p}, ignore_index=True)
                df.loc[len(df)-1, 'isSibling'] = True
    return df

def process_skills(df, name):
    df.dropna(subset=['we_c.vec','we_p.vec'], inplace=True)
    print(len(df))

    df['isSibling'] = False

    sub_category_nodes = df.groupby('p.name')['c.name'].unique().reset_index()
    df = append_category_nodes(df, sub_category_nodes)
    print(len(df))

    df['init_cosine'] = df.apply(lambda x: calc_cosine(x['we_c.vec'], x['we_p.vec']), axis=1)
    df.to_pickle(f'resources/skills_pickles/{name}.pkl')
    print(f'{name} is finished')


#APPLICATION_OF is ultra similar
#First group is most similar: leaf nodes that do not have any children. They are similar to other nodes that have a same BROADER skill and to their parent
query_leaves = '''
    MATCH (c:MSkill)-[:BROADER]->(p:MSkill)
    WHERE NOT (:MSkill)-[:BROADER]->(c)
    MATCH (c)-[we_c:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    MATCH (p)-[we_p:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN c.name, p.name, we_c.vec, we_p.vec'''
    

# df_leaves,r = runPy2neoQuery(neo4j,query_leaves)
# process_skills(df_leaves, 'first_group_skills')


#Second group is all other nodes in the middle: they have children and parents as well. They are similar to other nodes that have the same parent and to the parent itself.
query_middle = '''
    MATCH (:MSkill)-[:BROADER]->(c:MSkill)-[:BROADER]->(p:MSkill)
    WHERE (p)-[:BROADER]->(:MSkill)
    MATCH (c)-[we_c:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    MATCH (p)-[we_p:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN c.name, p.name, we_c.vec, we_p.vec
'''

# df_middle,r = runPy2neoQuery(neo4j,query_middle)
# process_skills(df_middle, 'second_group_skills')

#Third group is the least similar, it contains only broad parent nodes. They are similar to their children
query_root = '''
    MATCH (c:MSkill)-[:BROADER]->(p:MSkill)
    WHERE NOT (p)-[:BROADER]->(:MSkill)
    MATCH (c)-[we_c:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    MATCH (p)-[we_p:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN c.name, p.name, we_c.vec, we_p.vec
'''

# df_root,r = runPy2neoQuery(neo4j,query_root)
# process_skills(df_root, 'third_group_skills')

query_all_siblings = '''
    MATCH (c:MSkill)-[:BROADER]->(p:MSkill)
    WHERE NOT (c)-[:BROADER*2]->(:MSkill {name:'language'})
    MATCH (c)-[we_c:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    MATCH (p)-[we_p:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    RETURN c.name as child_name, p.name as parent_name , we_c.vec as child_vector, we_p.vec as parent_vector
'''

df_all_siblings,r = runPy2neoQuery(neo4j,query_all_siblings)
process_skills(df_all_siblings, 'all_skills_siblings')

query_all = '''
    MATCH (c:MSkill)-[we_c:CODED_IN]->(:SemanticModel{id:'GoogleWE'})
    WHERE NOT (c)-[:BROADER*2]->(:MSkill {name:'language'})
    RETURN c.name as skill, we_c.vec as vector
'''

df_all,r = runPy2neoQuery(neo4j,query_all)
df_all.to_pickle('resources/skills_pickles/all_skills.pkl')
print(f'all skills is finished')




