import os
import copy
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Optional, Any
from distributionalmodels.params import RNNParams, Params
from distributionalmodels.datasets.corpus_xAyBz import Corpus_xAyBz

random.seed(1023)


class SRN:
    # load srn from saved pretrained model
    @classmethod
    def from_pretrained(cls,
                        param_path: Path,
                        ):
        """Load RNN from saved state_dict"""

        # get params
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # init corpus
        corpus = Corpus_xAyBz(params.corpus_params.num_AB_categories,
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
                              'massed',
                              params.corpus_params.word_order_rule,
                              params.corpus_params.include_punctuation,
                              params.corpus_params.random_seed
                              )
        corpus.generate_paradigmatic_word_category_dict()
        dsm_list = []

        # load saved state_dict
        print(f'Looking for saved models in {param_path}')
        model_files = list(param_path.rglob('**/saves/*model.pt'))
        model_files = sorted(model_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))
        checkpoint_list = []
        print(f'Found {len(model_files)} saved models')
        for model_file in model_files:
            checkpoint = int(os.path.splitext(os.path.basename(model_file))[0].split('_')[1])
            if checkpoint not in checkpoint_list:
                checkpoint_list.append(checkpoint)
                state_dict = torch.load(model_file, map_location=torch.device('cpu'))
                dsm = cls(params.dsm_params, corpus)
                # # load state_dict into instance
                dsm.model.load_state_dict(state_dict)
                dsm.model.to(torch.device('cpu'))

                dsm_list.append(dsm)
            # print(f'Loaded model from {model_file}')

        return dsm_list, checkpoint_list, corpus

    # init function
    def __init__(self,
                 params: RNNParams,  # not everything is a float
                 corpus: Corpus_xAyBz,
                 save_path: Optional[Path] = None):

        self.params = params
        self.corpus = corpus
        self.save_path = save_path
        self.vocab_size = corpus.vocabulary_size
        self.vocab_id_dict = corpus.vocab_id_dict
        self.id_vocab_dict = corpus.id_vocab_dict
        self.encode = lambda s: [self.vocab_id_dict[v] for v in s]
        self.decode = lambda l: [self.id_vocab_dict[i] for i in l]
        self.numeric_document_list = copy.deepcopy(corpus.numeric_document_list)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = RNN()
        self.model.init_model(self.params, self.vocab_id_dict, self.device)

        self.performance = defaultdict(list)
        self.old_h = None

    # generate batches
    def gen_batches(self,
                    numeric_sequence_list: List[List[int]],
                    batch_size: Optional[int] = None,
                    shuffle: bool = False):

        print(f'Generating batches for {len(numeric_sequence_list)} sequences', flush=True)
        if batch_size is None:
            batch_size = self.params['batch_size']

        # shuffle and flatten   
        if shuffle:
            np.random.shuffle(numeric_sequence_list)

        is_leftover_batch = len(numeric_sequence_list) % batch_size != 0
        num_batches = len(numeric_sequence_list) // batch_size + int(is_leftover_batch)
        for i in range(num_batches):
            start = batch_size * i
            seq_b = numeric_sequence_list[start:start + batch_size]  # if end index is too large, small batch is created
            if len(seq_b) != batch_size:
                print(f'Found abnormal batch_size={len(seq_b)}')  # this is by design

            yield seq_b  # this is the batch used for training

    # evaluate function that aims to compute the perplexity from loss
    def calc_pp(self, numeric_document_list: List[List[int]]):

        print('Calculating perplexity...')

        self.model.eval()

        loss_total = 0
        num_loss = 0
        old_h = self.model.init_hidden_state()
        for document in numeric_document_list:
            for i in range(len(document[:-1])):
                input_vector = self.model.embed[document[i]]
                label_vector = self.model.embed[document[i + 1]]
                h, z_o, o_prob, loss = self.evaluate_item(input_vector, old_h, label_vector, self.optimizer)
                old_h = h.detach()
                loss_total += loss.item()
                num_loss += 1

        cross_entropy_average = loss_total / num_loss
        res = np.exp(cross_entropy_average)  # converting cross-entropy loss to perplexity score

        return res

    # training process for every epoch
    def train_epoch(self, numeric_document_list: List[List[int]]):
        self.model.train()

        old_h = self.model.init_hidden_state()
        # print(num_to_word(numeric_sequence_list, self.corpus.index_vocab_dict))
        for document in numeric_document_list:
            for i in range(len(document[:-1])):
                input_vector = self.model.embed[document[i]]
                label_vector = self.model.embed[document[i + 1]]
                h = self.train_item(input_vector, old_h, label_vector, self.optimizer)
                old_h = h.detach()

    # main training function
    def train(self,
              output_freq: int = 500):  # this number decides when to make an output
        self.model.to(self.device)  # move the model to whatever device you are using to train
        self.criterion = torch.nn.CrossEntropyLoss()  # loss function
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.params.learning_rate,
                                         momentum=self.params.momentum)  # optimizer

        # train loop
        save_index = 1
        pp_train = self.calc_pp(self.numeric_document_list)
        self.performance['epoch'].append(0)
        self.performance['pp_train'].append(pp_train)
        print(f'{0}: {pp_train:8.2f}')
        self.save_model(0, save_index)

        for epoch in range(1, self.params.num_epochs + 1):
            numeric_document_sentence_list = copy.deepcopy(self.corpus.numeric_document_sentence_list)
            final_numeric_document_sentence_list = []

            # shuffle the documents or sentences in documents according to the parameters in corpus
            if self.corpus.document_sequence_rule == 'random':
                random.shuffle(numeric_document_sentence_list)
                final_numeric_document_sentence_list = numeric_document_sentence_list
            if self.corpus.sentence_sequence_rule == 'random':
                for document in numeric_document_sentence_list:
                    random.shuffle(document)
                    final_numeric_document_sentence_list.append(document)
            if self.corpus.sentence_sequence_rule == 'massed':
                final_numeric_document_sentence_list = numeric_document_sentence_list

            final_numeric_document_list = \
                [[token for sentence in doc for token in sentence] for doc in final_numeric_document_sentence_list]

            # train on one epoch
            self.train_epoch(final_numeric_document_list)
            # switch between different output frequencies
            if epoch <= 100 or (100 < epoch <= 500 and epoch % 5 == 0) or (
                    500 < epoch <= 1000 and epoch % 10 == 0) or (
                    1000 < epoch <= 10000 and epoch % 25 == 0) or (
                    10000 < epoch <= 20000 and epoch % 50 == 0):
                pp_train = self.calc_pp(final_numeric_document_list)
                self.performance['epoch'].append(epoch)
                self.performance['pp_train'].append(pp_train)
                print(f'{epoch}: {pp_train:8.2f}')
                save_index += 1
                self.save_model(epoch, save_index)

        # evaluate the model at the end of the training processes
        pp_train = self.calc_pp(self.corpus.numeric_document_list)
        self.performance['pp_train'].append(pp_train)
        self.performance['epoch'].append(self.params.num_epochs + 1)
        print(f'last : {pp_train:8.2f}')
        self.save_model(epoch, save_index + 1)

    # getter function for performance dictionay
    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    # evaluate a single token
    def evaluate_item(self, x, old_h, y, torch_optimizer):
        z_h, h, z_o, o_prob = self.model.forward_item(x, old_h)
        torch_optimizer.zero_grad()
        index = torch.argmax(y).unsqueeze(dim=0)
        loss = self.criterion(z_o, index)
        return h, z_o, o_prob, loss

    # train a single token
    def train_item(self, x, old_h, y, torch_optimizer):
        z_h, h, z_o, o_prob = self.model.forward_item(x, old_h)
        torch_optimizer.zero_grad()
        index = torch.argmax(y).unsqueeze(dim=0)
        loss = self.criterion(z_o, index)
        loss.backward(retain_graph=True)
        torch_optimizer.step()
        return h

    # save the performance of the model to a csv file
    def save_performance(self, performance_list):
        file_location = 'models/' + self.name + '/performance.csv'
        outfile = open(file_location, 'a')
        output_string = ','.join(map(str, performance_list))
        outfile.write(output_string + '\n')
        outfile.close()

    # save the model as pickel file
    def save_model(self, current_epoch, save_index):
        torch.save(self.model.state_dict(), self.save_path / '{}_{}_model.pt'.format(save_index, current_epoch))


# class for pytorch RNN      
class RNN(nn.Module):
    def __init__(self):
        # many options here, like criterion optimizer are hard coded but need to be made params

        super().__init__()
        self.name = "SRN"
        self.start_datetime = None
        self.numeric_sequence_list = None
        self.vocab_dict = None
        self.vocab_size = None
        self.hidden_size = None
        self.weight_init_range = None

        self.input_size = None
        self.output_size = None
        self.current_epoch = None

        self.h_x = None  # weights from input to hidden
        self.y_h = None  # weights from hidden to output
        self.last_h = None
        self.sigmoid = None

        self.sigmoid = nn.Sigmoid().float()
        self.tanh = nn.Tanh().float()
        self.soft_max = nn.Softmax()

    #

    def init_model(self,
                   params: Dict[str, Any],
                   vocab_id_dict: Dict[str, int],
                   device):
        self.params = params
        self.vocab_id_dict = vocab_id_dict
        self.vocab_size = len(vocab_id_dict)
        self.hidden_size = params.embed_size
        self.learning_rate = params.learning_rate
        self.weight_init_range = params.embed_init_range

        self.input_size = self.vocab_size
        self.output_size = self.vocab_size
        self.device = device

        self.current_epoch = 0

        self.embed = F.one_hot(torch.tensor(list(vocab_id_dict.values())), num_classes=self.input_size)
        self.embed = self.embed.to(torch.float32).to(self.device)

        self.h_x = nn.Linear(self.input_size, self.hidden_size).float()
        self.h_h = nn.Linear(self.hidden_size, self.hidden_size).float()
        self.y_h = nn.Linear(self.hidden_size, self.output_size).float()

        self.h_x.apply(self.init_weights)
        self.h_h.apply(self.init_weights)
        self.y_h.apply(self.init_weights)

    def init_hidden_state(self):
        hidden = torch.zeros(self.hidden_size, dtype=torch.float32).to(self.device)
        return hidden

    def init_weights(self, m):
        m.weight.data.uniform_(-self.weight_init_range, self.weight_init_range)
        m.bias.data.uniform_(-self.weight_init_range, self.weight_init_range)

    def forward_item(self, x, old_h):
        z_h = self.h_x(x) + self.h_h(old_h)
        h = self.tanh(z_h)

        z_o = self.y_h(h)
        z_o = z_o.unsqueeze(dim=0)
        o_prob = torch.nn.functional.softmax(z_o, dim=1)
        return z_h, h, z_o, o_prob
