"""
This file defines the hyper-parameters used for each Ludwig job.
By collecting multiple values per hyper-parameter in a list in param2requests, 
Ludwig will run jobs corresponding to all combination of hyper-parameter values. 
Any hyper-parameter not overwritten by param2requests will be assigned its default value using param2default.

Notes: 
- Ludwig requires that the two dicts in this file be named as they are below. Do not rename them.
- Do not use non-hashable objects as values in param2requests or param2default.

"""
from dataclasses import dataclass, fields
from typing import Tuple, Optional


# submit jobs for one dsm at a time
DSM_NAME = ['ww_matrix',    # 0
            'wd_matrix',    # 1
            'srn',          # 2
            'lstm',         # 3
            'gpt',          # 4
            'w2v',          # 5
            ][4]

# will submit 3*2=6 jobs, each using a different learning rate and hidden sizes
param2requests = {
    # corpus params
    'num_AB_categories': [1],
    'AB_category_size': [2],
    'y_category_size': [1],


    # model params
    'num_epochs': [200],
    # 'learning_rate': [0.005],
    # 'embed_size' : [64],
    # 'embed_init_range' : [0.01],
    # 'momentum': [0.0],
    # 'random_seed': [1023]

    'round': [1]
}

if DSM_NAME == 'ww_matrix':
    param2default_dsm = {
        # window size means how many neighbors are considered in forward direction
        'count_type': ('ww', 'summed', 4, 'linear'),  # currently, sentence-boundary is respected automatically
        'norm_type': None,  # None is slightly better than all others
        'reduce_type': ('svd', 32),
    }

elif DSM_NAME == 'wd_matrix':
    param2default_dsm = {
        # window size means how many neighbors are considered in forward direction
        'count_type': ('wd', 'summed', 4, 'linear'),  # currently, sentence-boundary is respected automatically
        'norm_type': None,  # None is slightly better than all others
        'reduce_type': ('svd', 32),
    }
elif DSM_NAME == 'srn':
    param2default_dsm = {
        # architecture
        'rnn_type': 'srn',
        'embed_size': 8,  # 64 is better than any lower
        # optimization
        'embed_init_range': 0.1,  # 0.1 is good
        'num_epochs': 2000,  # more than 4 improves 1a and 2a accuracy, but 4 is best for 2b and 2c accuracy
        'learning_rate': 0.2,  # 0.06 with batch_size=64
        'momentum': 0.09,
        'round':1
    }

elif DSM_NAME == 'lstm':
    param2default_dsm = {
        # architecture
        'rnn_type': 'lstm',
        'embed_size': 9,            # 64 is better than any lower
        # optimization
        'embed_init_range': 0.001,    # 0.1 is good     # must be 0.0 with num_layers=1
        'num_epochs': 1000,            # more than 4 improves 1a and 2a accuracy, but 4 is best for 2b and 2c accuracy
        'learning_rate': 0.3,      # 0.06 with batch_size=64
        'momentum': 0.0,
        'round': 0

    }

elif DSM_NAME == 'w2v':
    param2default_dsm = {
        # architecture
        'embed_size': 9,
        'window_size': 2,
        # optimization
        'num_epochs': 10,
        'learning_rate': 0.025,
        'embed_init_range': 0.05,
        'round':0,
        'momentum': 0.0
    }
elif DSM_NAME == 'gpt':
    param2default_dsm = {
        'round': 0,
        'block_size': 5,
        'batch_size': 1,
        'num_epochs': 200,
        'embed_size': 64,
        'head_size': 64,
        'num_heads': 1,
        'learning_rate': 0.0005
    }
# [B13 . A11 y2] B12
# Draw picture of connection

else:
    raise NotImplementedError

param2requests['dsm'] = [DSM_NAME]

param2default = {
    'dsm': None,
    'composition_fn': 'native',
}
# the bigger the category omitted b should be confused with legal b

param2default_corpus = {
    'num_AB_categories': 2,
    'AB_category_size': 3,

    'x_category_size':0,
    'y_category_size':3,
    'z_category_size':0,

    'min_x_per_sentence':0,
    'max_x_per_sentence':0,
    'min_y_per_sentence':1,
    'max_y_per_sentence':1,
    'min_z_per_sentence':0,
    'max_z_per_sentence':0,

    'document_organization_rule':'all_pairs',
    'document_repetitions' : 1,
    'document_sequence_rule':'massed',

    'sentence_repetitions_per_document': 0,
    'sentence_sequence_rule':'massed',

    'word_order_rule': 'fixed',
    'include_punctuation': True,

    'random_seed': None
}

for k in param2default_corpus:
    assert k not in param2default_dsm
param2default.update(param2default_corpus)
param2default.update(param2default_dsm)


@dataclass
class CorpusParams:
    num_AB_categories: int
    AB_category_size: int
    x_category_size: int
    y_category_size: int
    z_category_size: int
    min_x_per_sentence: int
    max_x_per_sentence: int
    min_y_per_sentence: int
    max_y_per_sentence: int
    min_z_per_sentence: int
    max_z_per_sentence: int
    document_organization_rule: str
    document_repetitions: int
    document_sequence_rule: str
    sentence_repetitions_per_document: int
    sentence_sequence_rule: str
    word_order_rule: str
    include_punctuation: bool
    random_seed: int

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

class WWParams:
    count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    # ('ww', 'concatenated',  4,  'linear')
    # ('wd', None, None, None)
    norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'
    reduce_type: Tuple[Optional[str], Optional[int]]  # e.g. ('svd', 200) or (None, None)

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

class WDParams:
    count_type: Tuple[str, Optional[str], Optional[int], Optional[str]]
    # ('ww', 'concatenated',  4,  'linear')
    # ('wd', None, None, None)
    norm_type: Optional[str]  # e.g. None, 'row_sum', 'row_logentropy', 'tf_idf', 'ppmi'
    reduce_type: Tuple[Optional[str], Optional[int]]  # e.g. ('svd', 200) or (None, None)

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

@dataclass
class RNNParams:
    # architecture
    rnn_type: str
    embed_size: int
    # optimization
    embed_init_range: float
    num_epochs: int
    learning_rate: float
    momentum: float
    round: int

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

@dataclass
class LSTMParams:
    # architecture
    rnn_type: str
    embed_size: int
    # optimization
    embed_init_range: float
    num_epochs: int
    learning_rate: float
    momentum: float
    round: int
    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

@dataclass
class Word2VecParams:
    embed_size: int
    window_size: int
    num_epochs: int
    learning_rate: int
    embed_init_range: float
    round: int
    momentum: float

    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

@dataclass
class GPTParams:
    round: int
    block_size: int
    batch_size: int
    num_epochs: int
    embed_size: int
    head_size: int
    num_heads: int
    learning_rate: float
    
    @classmethod
    def from_param2val(cls, param2val):
        field_names = set(f.name for f in fields(cls))
        return cls(**{k: v for k, v in param2val.items() if k in field_names})

@dataclass
class Params:

    corpus_params: CorpusParams
    dsm: str
    dsm_params: WWParams
    composition_fn: str
    composition_fn: str

    @classmethod
    def from_param2val(cls, param2val):

        # exclude keys from param2val which are added by Ludwig.
        # they are relevant to job submission only.
        tmp = {k: v for k, v in param2val.items()
               if k not in ['job_name', 'param_name', 'save_path', 'project_path']}

        if param2val['dsm'] == 'count':
            dsm_params = WWParams.from_param2val(tmp)
        elif param2val['dsm'] == 'random':
            dsm_params = WDParams.from_param2val(tmp)
        elif param2val['dsm'] == 'w2v':
            dsm_params = Word2VecParams.from_param2val(tmp)
        elif param2val['dsm'] == 'srn':
            dsm_params = RNNParams.from_param2val(tmp)
        elif param2val['dsm'] == 'lstm':
            dsm_params = LSTMParams.from_param2val(tmp)
        elif param2val['dsm'] == 'gpt':
            dsm_params = GPTParams.from_param2val(tmp)
        else:
            raise AttributeError(f'Invalid arg to "dsm" "{param2val["dsm"]}".')

        corpus_params = CorpusParams.from_param2val(tmp)
        return cls(dsm=param2val['dsm'],
                   composition_fn=param2val['composition_fn'],
                   corpus_params=corpus_params,
                   dsm_params=dsm_params,
                   )