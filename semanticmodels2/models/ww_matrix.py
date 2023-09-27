import numpy as np
from cytoolz import itertoolz
from typing import List, Tuple, Dict
from semanticmodels2.models.count_matrix import CountMatrix

PAD = '*PAD*'
VERBOSE = False

class WWMatrix(CountMatrix):

    def __init__(self, 
                 vocab_dict: Dict[str, int],
                 numeric_sequence_list: List[List[int]],
                 window_size=7, 
                 window_direction='forward', 
                 window_weighting='flat'):
        super().__init__(vocab_dict, numeric_sequence_list)

        self.window_size = window_size
        self.window_direction = window_direction
        self.window_weighting = window_weighting
    
    def __repr__(self):
        output_string = "\n"
        output_string += "WW Matrix\n"
        output_string += "    Num Documents: {}\n".format(self.num_docs)
        output_string += "    Num Words: {}\n".format(self.vocab_size)
        output_string += "    Window Size: {}\n".format(self.window_size)
        output_string += "    Window Direction: {}\n".format(self.window_direction)
        output_string += "    Window Weighting: {}\n".format(self.window_weighting)

        return output_string

    def create_ww_matrix(self):
        self.num_docs = len(self.numeric_sequence_list)
        self.count_matrix = np.zeros([self.vocab_size, self.vocab_size], int)
        temp_count_matrix = np.zeros([self.vocab_size, self.vocab_size], int)
        for token_ids in self.numeric_sequence_list:
            token_ids += [PAD] * self.window_size  # add padding such that all co-occurrences in last window are captured

            if VERBOSE:
                print(token_ids)
            for w in itertoolz.sliding_window(self.window_size + 1, token_ids):  # + 1 because window consists of t2s only
                if VERBOSE:
                    print([list(self.vocab_dict.keys())[list(self.vocab_dict.values()).index(i)] if isinstance(i, int) else PAD for i in w])
                t1_id = w[0]
                for dist, t2_id in enumerate(w[1:]):

                    # increment
                    if t1_id == PAD or t2_id == PAD:
                        continue
                    if self.window_weighting == "linear":
                        temp_count_matrix[t1_id, t2_id] += self.window_size - dist
                    elif self.window_weighting == "flat":
                        temp_count_matrix[t1_id, t2_id] += 1
                    if VERBOSE:
                        print('row {:>3} col {:>3} set to {}'.format(t1_id, t2_id, self.count_matrix[t1_id, t2_id]))

        # window_type
        if self.window_direction == 'forward':
            self.count_matrix = temp_count_matrix
        elif self.window_direction == 'backward':
            self.count_matrix = temp_count_matrix.transpose()
        elif self.window_direction == 'summed':
            self.count_matrix = temp_count_matrix + temp_count_matrix.transpose()
        elif self.window_direction == 'concatenated':
            self.count_matrix = np.concatenate((temp_count_matrix, temp_count_matrix.transpose()), axis=1)
        else:
            raise AttributeError('Invalid arg to "window_type".')

        print('Shape of normalized matrix={}'.format(self.count_matrix.shape))