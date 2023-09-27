import numpy as np
from scipy.spatial import distance
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd


def print_neighbhors_from_file():
    similarity_file = "results/csv_files/ww_similarities.csv"
    f = open(similarity_file)
    header_list = []
    data_list = []

    i = 0
    for line in f:
        if i > 0:
            data = (line.strip().strip('\n').strip()).split(',')
            header_list.append(data[0])
            data_list.append(data[1:])
        i += 1
    f.close()

    data_matrix = np.array(data_list, float)
    print(data_matrix)
    num_words = len(header_list)

    for i in range(num_words):
        current_sims = data_matrix[:,i]
        sorted_indexes = np.argsort(current_sims)
        print(header_list[i])
        for j in range(num_words):
            current_index = sorted_indexes[j]
            print(header_list[current_index], current_sims[current_index])
        print()


def create_similarity_matrix(embedding_matrix, metric:str):
    if metric == "corrcoef":
        similarity_matrix = np.corrcoef(embedding_matrix)
    elif metric == "cosine":
        similarity_matrix = distance.squareform(distance.pdist(embedding_matrix, "cosine"))
    return similarity_matrix

def print_neighbors_from_embeddings(embedding_matrix, vocab_dict, metric: str = "corrcoef", verbose: bool = False):
    if metric == "corrcoef":
        dist_matrix = np.corrcoef(embedding_matrix)
    elif metric == "cosine":
        dist_matrix = distance.squareform(distance.pdist(embedding_matrix, "cosine"))
    vocab_size = len(vocab_dict)
    for word1, index1 in vocab_dict.items():
        data_list = []
        for word2, index2 in vocab_dict.items():
            if word1 != word2:
                data_list.append((dist_matrix[index1, index2], word2))
        
        data_list.sort(key = lambda x: x[0], reverse=True)
        if verbose: 
            print(word1)

            for i in range(vocab_size-1):
                print("    {}: {:0.3f}".format(data_list[i][1], data_list[i][0]))

    return dist_matrix

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler

def num_to_word(list, dict):
    new_doc_list = []
    for doc in list:
        new_doc_list.append([dict[num] for num in doc])
    return new_doc_list

def round_and_check(x, place):
    if pd.isna(x) or not np.isscalar(x) or not np.isreal(x):
        return x
    else:
        return round(x, place)
    

