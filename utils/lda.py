import os
import gensim
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import  numpy as np

from utils.helper_functions import *
from utils.evaluation import *

def grid_search(K, dir_priors, random_states, num_passes, iterations, save_folder, bow_corpus, dictionary):
    '''
    Perform a grid search of LDA model  (topic modelling)

    Keyword arguments:
    K (list of int) - Numbers of requested latent topics to be extracted from the training corpus
    dir_priors (list of str)  - Default prior selecting strategies for topics and words, can be ’symmetric’, 
        ’asymmetric’ or ’auto’
    random_states (list of int) - List of seeds to generate random state
    num_passes (list of int) - Number of passes through the corpus during training
    iterations (list of int) - Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
    save_folder (str) -  A path to the folder in which the LDA models will be stored
    bow_corpus (list of tuples) - Stream of document vectors or sparse matrix of shape (num_documents, num_terms)
    dictionary (dict of (int, str)) - Mapping from word IDs to words. It is used to determine the vocabulary size

    '''

    for k in K:
        for dir_prior in dir_priors:
            for random_state in random_states:
                for num_pass in num_passes:
                    for iteration in iterations:
                        # check if model already created
                        target_folder = os.path.join(save_folder, str(k), dir_prior, str(random_state), str(num_pass), str(iteration))
                        print(target_folder)
                        if not (os.path.exists(target_folder)):
                            # create target folder to save LDA model to
                            create_directory(target_folder)
                            model = gensim.models.LdaModel(bow_corpus, 
                                                            id2word = dictionary,
                                                            num_topics = k,
                                                            iterations= iteration, 
                                                            passes = num_pass, 
                                                            minimum_probability = 0, 
                                                            alpha = dir_prior, 
                                                            eta = dir_prior, 
                                                            eval_every = None,
                                                            random_state= random_state)
                            # save LDA model
                            model.save(os.path.join(target_folder, 'lda.model'))
                        else:
                            print('LDA model already exists, skipping ...')


def get_coherence_scores(save_folder, company_df, dictionary):
    '''
    Get coherence scores for the LDA models and save the scores

    Keyword arguments:
    save_folder (str) -  A path to the folder in which the LDA models are stored
    company_df (dataframe) - A dataframe containing tokenized texts (a column 'tokens')
    dictionary (dict of (int, str)) - Mapping from word IDs to words. It is used to determine the vocabulary size

    '''
    M = [x for x in read_directory(save_folder) if x.endswith('lda.model')]
    scores = []

    for i, m in enumerate(M):
        print(m)
        # number of topics
        k = m.split(os.sep)[2]
        # different dirichlet priors
        dir_prior = m.split(os.sep)[3]
        # random initiatilizations
        random_state = m.split(os.sep)[4]
        # passes over the corpus
        num_pass = m.split(os.sep)[5]
        # max iteration for convergence
        iteration = m.split(os.sep)[6]
        model = gensim.models.LdaModel.load(m)
        coherence_c_v = CoherenceModel(model = model, texts = company_df['tokens'].tolist(), dictionary = dictionary, coherence='c_v')
        score = coherence_c_v.get_coherence()
        print(f'coherence score: {score}')
        doc = {	'k' : k, 'dir_prior' : dir_prior, 'random_state' : random_state, 'num_pass' : num_pass, 'iteration' : iteration, 'coherence_score' : score}
        scores.append(doc)

    data = [[int(x['k']), x['dir_prior'],x['random_state'], x['num_pass'], x['iteration'], x['coherence_score']] for x in scores]
    df = pd.DataFrame()
    for k in range(2, 20 + 1):
        # create dataframe to temporarily store values
        df_temp = pd.DataFrame(index = [k])

        # loop trough the dat a to obtain only the scores for a specific k value
        for row in sorted(data):
            if row[0] == k:
                df_temp['{}-{}-{}-{}'.format(row[1],row[2],row[3],row[4])] = pd.Series(row[5], index=[k])

        # append temporarary dataframe of only 1 k value to the full dataframe 
        df = df.append(df_temp)
    if not (os.path.exists(os.path.join(save_folder, 'results'))):
        create_directory(os.path.join(save_folder, 'results'))
    df.to_csv(os.path.join(save_folder, 'results', 'results_coherence.csv'))
    print(f'Scores are stored in {save_folder}/results/results_coherence.csv')


def output_lda_topics(K = 6, dir_prior = 'auto', random_state = 99, num_pass = 5, iteration = 200, top_n_words = 10, models_folder = 'LDA/telekom/', save_folder = 'LDA/results/telekom'):
    '''
    Print the topics for the best chosen LDA model (top n words for each topic) and save it to the csv file

    Keyword arguments:
    K (int) - Number of requested latent topics
    dir_prior (str)  - Default prior selecting strategies for topics and words, can be ’symmetric’, 
        ’asymmetric’ or ’auto’
    random_state (int) - Seed to generate random state
    num_pass (int) - Number of passes through the corpus during training
    iteration (int) - Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
    top_n_words (int) -The number of words to print for each topic
    models_folder (str) - A path to the folder where LDA models are stored
    save_folder (str) -  A path to the folder where scored results are stored
    '''

    best_model = os.path.join(models_folder, str(K), dir_prior, str(random_state), str(num_pass), str(iteration), 'lda.model')
    print(best_model)
    model = gensim.models.LdaModel.load(best_model)

    # define empty lists so we can fill them with words		
    topic_table, topic_list = [], []

    # loop trough all the topics found within K
    for k in range(K):
        # add column for word and probability
        topic_table.append(["word", "prob."])

        list_string = ""
        topic_string = ""
        topic_string_list = []

        # get topic distribution for topic k and return only top-N words 
        scores = model.print_topic(k, top_n_words).split("+")

        # loop trough each word and probability
        for score in scores:

            # extract score and trimm spaces
            score = score.strip()

            # split on *
            split_scores = score.split('*')

            # get percentage
            percentage = split_scores[0]
            # get word
            word = split_scores[1].strip('"')

            # add word and percentage to table
            topic_table.append([word.upper(), "" + percentage.replace("0.", ".")])
            
            # add word to list table
            list_string += word + ", "

        # add empty line for the table
        topic_table.append([""])
        # add topic words to list
        topic_list.append([str(k+1), list_string.rstrip(", ")])

    for topic in topic_list:
        print(f'topic {topic[0]}: {topic[1:]}')
    # save to CSV
    save_csv(topic_list, 'topic-list', folder = save_folder)
    save_csv(topic_table, 'topic-table', folder = save_folder)

def infer_doc_topics(dictionary, corpus, df_desc, K = 5, dir_prior = 'auto', random_state = 42, num_pass = 20, iteration = 200, top_n_words = 10, models_folder = 'LDA/', save_folder = 'LDA/results/'):
    '''
    Infer a topic probability for each doc

    Keyword arguments:
    dictionary (dict of (int, str)) - Mapping from word IDs to words. It is used to determine the vocabulary size
    corpus (list of tuples) - Stream of document vectors or sparse matrix of shape (num_documents, num_terms)
    df_desc (dataframe) - A dataframe containing tokenized texts (a column 'tokens')
    K (int) - Number of requested latent topics
    dir_prior (str)  - Default prior selecting strategies for topics and words, can be ’symmetric’, 
        ’asymmetric’ or ’auto’
    random_state (int) - Seed to generate random state
    num_pass (int) - Number of passes through the corpus during training
    iteration (int) - Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
    top_n_words
    models_folder (str) - A path to the folder where LDA models are stored
    save_folder (str) -  A path to the folder where scored results are stored

    Returns a dataframe with 'topics' column
    '''

    # load LDA model according to parameters
    best_model = os.path.join(models_folder, str(K), dir_prior, str(random_state), str(num_pass), str(iteration), 'lda.model')
    print(best_model)
    model = gensim.models.LdaModel.load(best_model)
    df_desc['topics'] = np.empty((len(df_desc), 0)).tolist()

    for i, row in df_desc.iterrows():
        print(f'Processing job {i}')
        bow = model.id2word.doc2bow(row['tokens'])
        topics = model.get_document_topics(bow, per_word_topics = False)
        topics = [y for x, y in topics]
        df_desc.at[i, 'topics'] = topics
    return df_desc

def get_doc_similarity(company_df, path_to_gold_standard):
    num_rows = company_df.shape[0]
    data = np.zeros((num_rows, num_rows))
    # loop over rows in dataframe
    for i in range(0, num_rows):

        # loop over the same rows again so we can compare them
        for j in range(0, num_rows):
            # get values from row i
            row_i = company_df.iloc[i]['topics']
            # get values from row j
            row_j = company_df.iloc[j]['topics']
            # calculate hellinger distance (lower scores are more similiar)
            hellinger_distance = calculate_hellinger_distance(row_i, row_j)

            # add to data
            data[i,j] = hellinger_distance
    print('Hellinger distance is computed')
    gold_standard = pd.read_excel(path_to_gold_standard, dtype=str)

    gold_standard_dict = {}
    for i, row in gold_standard.iterrows():
        similar_jobs = row.filter(regex=("sim.*")).dropna().to_list()
        similar_job_indexes = [gold_standard[gold_standard['id']==_id].index[0] for _id in similar_jobs]
        gold_standard_dict[i] = similar_job_indexes

    map_values = []
    re_values = []
    map_thresh = []
    for thresh in np.arange(0, 1.05, 0.05):
        predictions = get_predicted_results(gold_standard_dict, data, thresh, reverse=True)
        map_precision, recall = get_map(gold_standard_dict, predictions)
        map_thresh.append(thresh)
        map_values.append(map_precision)
        re_values.append(recall)

    return map_values, re_values, map_thresh, gold_standard, data