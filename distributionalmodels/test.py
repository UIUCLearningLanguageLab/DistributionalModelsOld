import pandas as pd
from typing import Dict, List
from models.count_matrix import CountMatrix
from datasets.test_dataset import TestDataset


def main():
# loading a dataset
    #the_dataset = AOChildes()
    the_dataset = TestDataset(num_documents=10, sent_per_doc=5)
    numeric_token_sequence_List = the_dataset.numeric_token_sequence_list

    # init model
    ww_count_matrix = CountMatrix(the_dataset.vocab_dict, numeric_token_sequence_List)
    wd_count_matrix = CountMatrix(the_dataset.vocab_dict, numeric_token_sequence_List)

    # train model
    ww_count_matrix.create_ww_matrix_fast()
    print(numeric_token_sequence_List)
    ww_count_matrix.reduce_svd(10)
    ww_count_matrix.compute_similarity_matrix(ww_count_matrix.svd_reduced_matrix, 'pearsonr')
    

    wd_count_matrix.create_wd_matrix()
    wd_count_matrix.reduce_svd(10)
    wd_count_matrix.compute_similarity_matrix(wd_count_matrix.svd_reduced_matrix, 'pearsonr')


main()

    