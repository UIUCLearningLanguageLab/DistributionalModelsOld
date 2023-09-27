import numpy as np
from typing import List, Tuple, Dict
from semanticmodels2.models.count_matrix import CountMatrix


class WDMatrix(CountMatrix):

    def __init__(self, 
                 vocab_dict: Dict[str, int],
                 numeric_sequence_list: List[List[int]]):
        super().__init__(vocab_dict, numeric_sequence_list)
    
    def __repr__(self):
        output_string = "\n"
        output_string += "WD Matrix\n"
        output_string += "    Num Documents: {}\n".format(self.num_docs)
        output_string += "    Num Words: {}\n".format(self.vocab_size)
        return output_string

    
    def create_wd_matrix(self):
        self.num_docs = len(self.numeric_sequence_list)
        self.count_matrix = np.zeros([self.vocab_size, self.num_docs], int)
        print('\nCounting word occurrences in {} documents'.format(self.num_docs))
        for i in range(self.num_docs):
            for j in self.numeric_sequence_list[i]:
                self.count_matrix[j, i] += 1