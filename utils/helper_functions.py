import os, sys, logging, glob2, csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_directory(name):
    """
    Create directory if not exists
    Parameters
    ----------
    name : string
            name of the folder to be created
    """
    try:
        if not os.path.exists(name):
            os.makedirs(name)
            logging.info('Created directory: {}'.format(name))
    except Exception(e):
        logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))


def read_directory(directory):
    """
    Read file names from directory recursively
    Parameters
    ----------
    directory : string
                directory/folder name where to read the file names from
    Returns
    ---------
    files : list of strings
            list of file names
    """

    try:
        return glob2.glob(os.path.join(directory, '**' , '*.*'))
    except Exception(e):
        logging.error("[{}] : {}".format(sys._getframe().f_code.co_name,e))

def save_csv(data, name, folder):

    """
    Save list of list as CSV (comma separated values)
    Parameters
    ----------
    data : list of list
            A list of lists that contain data to be stored into a CSV file format
    name : string
            The name of the file you want to give it
    folder: string
            The folder location
    """

    try:

    # create folder name as directory if not exists
        create_directory(folder)

        # create the path name (allows for .csv and no .csv extension to be handled correctly)
        suffix = '.csv'
        if name[-4:] != suffix:
            name += suffix

        # create the file name
        path = os.path.join(folder, name)

        # save data to folder with name
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(data)

    except Exception(e):
        logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))

def calculate_hellinger_distance(p, q):

    """
        Calculate the hellinger distance between two probability distributions
        note that the hellinger distance is symmetrical, so distance p and q = q and p
        other measures, such as KL-divergence, are not symmetric but can be used instead
        Parameters
        -----------
        p : list or array
            first probability distribution
        q : list or array
            second probability distribution
        Returns
        --------
        hellinger_dinstance : float
            hellinger distance of p and q
    """

    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def calc_cosine(vec_a, vec_b):
    return cosine_similarity([vec_a], [vec_b])[0][0]


def read_vectors(file):

    attract_repel_skills = pd.read_csv(file, sep=" ", header=None)
    attract_repel_skills['vector'] = attract_repel_skills[attract_repel_skills.columns[1:]].values.tolist()
    attract_repel_skills.rename(columns={0:'id'}, inplace=True)
    attract_repel_skills = attract_repel_skills[['id', 'vector']].set_index('id')

    return attract_repel_skills