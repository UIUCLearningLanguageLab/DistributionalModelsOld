from semanticmodels2.params import Params
import pandas as pd
import numpy as np
from typing import Dict, List
from semanticmodels2.models.wd_matrix import WDMatrix
from semanticmodels2.models.ww_matrix import WWMatrix
from semanticmodels2.models.lstm import LSTM
from semanticmodels2.models.srn import SRN
from semanticmodels2.models.gpt import GPT
from semanticmodels2.models.word2vec import W2Vec, Word2VecDataset
from semanticmodels2.datasets.test_dataset import TestDataset
from semanticmodels2.datasets.corpus_xAyBz import Corpus_xAyBz
# from tasks.result import Result
# from tasks.cohyponym_task import CohyponymTask
# from tasks.syntagmatic_relations import SyntagmaticRelations
# from scripts.utils import print_neighbors_from_embeddings
# from scripts.visualization import visualization_with_tsne, plot_confusion_matrix, plot_balanced_accuracy
import os
import time
import torch
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)


def main(param2val):
    """This function is run by Ludwig on remote workers."""
    params = Params.from_param2val(param2val)
    if Path(param2val['save_path']).exists():
        save_path = Path(param2val['save_path'])
    else:
        path = param2val['save_path']
        split_path = path.split("semanticmodels2_", 1)
        save_path = Path(split_path[1]) if len(split_path) > 1 else None
        os.makedirs(save_path, exist_ok=True)

        # Check if there's a result after the split and extract it
        result = split_path[1] if len(split_path) > 1 else None

    # save_path = Path('datasets')
    # the_dataset = TestDataset("Test1", num_documents=10, sent_per_doc=1000)
    the_corpus = Corpus_xAyBz(params.corpus_params.num_AB_categories,
                              params.corpus_params.AB_category_size,
                              params.corpus_params.x_category_size,
                              params.corpus_params.y_category_size,
                              params.corpus_params.z_category_size,
                              params.corpus_params.min_x_per_sentence,
                              params.corpus_params.max_x_per_sentence,
                              params.corpus_params.min_y_per_sentence,
                              params.corpus_params.max_y_per_sentence,
                              params.corpus_params.min_z_per_sentence,
                              params.corpus_params.max_z_per_sentence,
                              params.corpus_params.document_organization_rule,
                              params.corpus_params.document_repetitions,
                              params.corpus_params.document_sequence_rule,
                              params.corpus_params.sentence_repetitions_per_document,

                              params.corpus_params.sentence_sequence_rule,
                              params.corpus_params.word_order_rule,
                              params.corpus_params.include_punctuation,
                              params.corpus_params.random_seed
                              )

    if not Path('corpus_info').exists():
        Path('corpus_info').mkdir(parents=True)
    corpus_info_path = Path("corpus_info/{}_corpus".format(the_corpus.corpus_name))
    if not corpus_info_path.exists():
        os.makedirs(corpus_info_path, exist_ok=True)

    the_corpus.export_corpus_as_file(corpus_info_path)
    the_corpus.create_paradigmatic_category_file(corpus_info_path)
    the_corpus.create_syntagmatic_category_file(corpus_info_path)

    if params.dsm == 'ww_matrix':
        dsm = WWMatrix(the_corpus.vocab_id_dict, the_corpus.numeric_document_list, window_size=7)
    elif params.dsm == 'wd_matrix':
        dsm = WDMatrix(the_corpus.vocab_id_dict, the_corpus.numeric_document_list)
    elif params.dsm == 'w2v':
        dsm = W2Vec(params.dsm_params, the_corpus, save_path)
    elif params.dsm == 'srn':
        dsm = SRN(params.dsm_params, the_corpus, save_path)
    elif params.dsm == 'lstm':
        dsm = LSTM(params.dsm_params, the_corpus, save_path)
    elif params.dsm == 'gpt':
        dsm = GPT(params.dsm_params, the_corpus, save_path)
    else:
        raise NotImplementedError
    series_list = []

    train_function = getattr(dsm, 'train', None)
    if callable(train_function):
        dsm.train()
        print(f'Completed training the DSM', flush=True)

        performance = dsm.get_performance()

        for k, v in performance.items():
            if k == 'epoch':
                continue
            s = pd.Series(v, index=performance['epoch'])
            s.name = k
            series_list.append(s)

            # for each save_epoch:
            #   save a pickle file of the model
            # save the model as pickel file

    print('Completed main.job.', flush=True)

    return series_list
