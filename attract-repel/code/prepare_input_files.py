import pandas as pd



def read_process(pickle_path):
    df = pd.read_pickle(pickle_path)
    df['skill'] = df['skill'].str.replace('’', '\'').str.replace(' ', '_')

    return df

def write_vectors(df):
    with open('attract_repel/word-vectors/init_google_we.txt', 'a') as we_file:
        for i, row in df.iterrows():
            try:
                we_file.write(f'{row["skill"]} {" ".join(str(item) for item in row["vector"])}\n')
            except:
                print('error')
                print(row['skill'])

    print('initial vectors written')

def write_similar_skills(df, name):
    with open(f'attract_repel/linguistic_constraints/{name}.txt', 'a') as sim_file:
        for i, row in df.iterrows():
            try:
                sim_file.write(f'{row["c.name"]} 	{row["p.name"]}\n')
            except:
                print('error')
                print(row['c.name'], row['p.name'])
    print(f'{name} is written')

if __name__ == "__main__":
    
    read_process('resources/skills_pickles/all_skills.pkl')

    write_vectors(df_skills)

    write_similar_skills(df_skills, 'similar_skills_all')