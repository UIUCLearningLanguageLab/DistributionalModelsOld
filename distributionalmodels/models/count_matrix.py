import numpy as np
from typing import List, Tuple, Dict
import copy
PAD = '*PAD*'
VERBOSE = False

class CountMatrix:
    def __init__(self,
                 vocab_dict: Dict[str, int],
                 numeric_sequence_list: List[List[int]]):

        self.numeric_sequence_list = copy.deepcopy(numeric_sequence_list)
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.num_docs = None

        self.count_matrix = None

        self.singular_values = None
        self.row_singular_values = None
        self.column_singular_values = None

        self.similarity_matrix = None

    def reduce_svd(self, num_dimensions=0):
        ## TODO move this and compute similarity from here to a more general utils.py file
        self.row_singular_values, self.singular_values, self.column_singular_values = np.linalg.svd(self.count_matrix)
        
        # to calculate num_dimensions automatically, pick a value where that value has 75% of the variance

        if num_dimensions == 0:
            s_probs = self.singular_values/sum(self.singular_values)
            value = 0
            counter = 0
            while value < .75:
                value += s_probs[counter]
                counter += 1

            # do some error checking to make sure that num_dimensions <= length of s
        if num_dimensions > len(self.singular_values):
            raise Exception("ERROR: Lenth of num_dimensions is greater than the lenth of the s vector")

        self.svd_reduced_matrix = self.row_singular_values[:,:num_dimensions]
    
    def compute_similarity_matrix(self, input_matrix, similarity_metric):
        # output_string = ""
        # for i in range(len(input_matrix[:,0])):
        #     for j in range(len(input_matrix[0,:])):
        #         output_string += " {0.3f}".format(i,j)


        if similarity_metric == 'pearsonr':
            self.similarity_matrix = np.corrcoef(input_matrix)
        else:
            raise Exception("ERROR: Unrecognized similarity metric {}".format(similarity_metric))

        # TODO consider other faster implimentations
        # metrics = correlation, cosine, distance
        