import numpy as np
from typing import List, Dict, Optional
from distributionalmodels.params import LSTMParams, Params
from pathlib import Path
import torch
import copy
import random
from distributionalmodels.datasets.corpus_xAyBz import Corpus_xAyBz
import yaml
from collections import defaultdict
import os

random.seed(1023)


class LSTM:
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
                              'random',
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

    def __init__(self,
                 params: LSTMParams,  # not everything is a float
                 corpus: Corpus_xAyBz,
                 save_path: Optional[Path] = None,
                 ):
        self.params = params
        self.corpus = corpus
        self.save_path = save_path
        self.vocab_size = self.corpus.vocabulary_size
        self.vocab_id_dict = corpus.vocab_id_dict
        self.id_vocab_dict = corpus.id_vocab_dict
        self.numeric_document_list = copy.deepcopy(self.corpus.numeric_document_list)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = MODEL(self.params, self.vocab_size, self.device)

        self.performance = defaultdict(list)
        self.old_h = None

    def train_item(self, input, label, h, c):
        z_o, o_prob, h, c = self.model(input, h, c)
        self.optimizer.zero_grad()
        label_tensor = torch.tensor(label).to(self.device)
        loss = self.criterion(z_o, label_tensor)
        loss.backward()
        self.optimizer.step()
        return h, c

    def test_item(self, input, label, h, c):
        z_o, o_prob, h, c = self.model(input, h, c)
        self.optimizer.zero_grad()
        label_tensor = torch.tensor(label).to(self.device)
        loss = self.criterion(z_o, label_tensor)
        o = self.model.sigmoid(z_o)
        return o_prob, loss, h, c

    def calc_pp(self, numeric_document_list: List[List[int]]):
        print('Calculating perplexity...')

        self.model.eval()
        loss_total = 0
        num_loss = 0
        old_h, old_c = self.model.init_hidden_state()
        for document in numeric_document_list:
            for i in range(len(document[:-1])):
                o_prob, loss, h, c = self.test_item([document[i]], [document[i + 1]], old_h, old_c)
                old_h, old_c = h.detach(), c.detach()
                loss_total += loss.item()
                num_loss += 1

        cross_entropy_average = loss_total / num_loss
        res = np.exp(cross_entropy_average)  # converting cross-entropy loss to perplexity score
        return res

    def train_epoch(self, numeric_document_list: List[List[int]]) -> None:
        self.model.train()
        old_h, old_c = self.model.init_hidden_state()
        for document in numeric_document_list:
            for i in range(len(document[:-1])):
                h, c = self.train_item([document[i]], [document[i + 1]], old_h, old_c)
                old_h, old_c = h.detach(), c.detach()

    def train(self, output_freq: int = 1000):
        self.model.to(self.device)  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()  # loss function
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.learning_rate)

        save_index = 1
        pp_train = self.calc_pp(self.numeric_document_list)
        self.performance['epoch'].append(0)
        self.performance['pp_train'].append(pp_train)
        print(f'{0}: {pp_train:8.2f}')

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
            if epoch <= 100 or (100 < epoch <= 500 and epoch % 5 == 0) or (
                    500 < epoch <= 1000 and epoch % 10 == 0) or (
                    1000 < epoch <= 10000 and epoch % 25 == 0) or (
                    10000 < epoch <= 20000 and epoch % 50 == 0):
                # if epoch % output_freq == 0:
                pp_train = self.calc_pp(final_numeric_document_list)
                self.performance['epoch'].append(epoch)
                self.performance['pp_train'].append(pp_train)
                print(f'{epoch}: {pp_train:8.2f}')
                save_index += 1
                self.save_model(epoch, save_index)

        pp_train = self.calc_pp(self.numeric_document_list)
        self.performance['pp_train'].append(pp_train)
        self.performance['epoch'].append(self.params.num_epochs + 1)
        print(f'last : {pp_train:8.2f}')

        self.save_model(epoch, save_index + 1)

    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    def save_model(self, current_epoch, save_index):
        torch.save(self.model.state_dict(), self.save_path / '{}_{}_model.pt'.format(save_index, current_epoch))


class MODEL(torch.nn.Module):
    def __init__(self,
                 params: LSTMParams,
                 vocab_size: int,
                 device
                 ):
        super().__init__()
        self.embed_size = params.embed_size
        self.embed_init_range = params.embed_init_range
        self.hidden = None
        self.device = device
        self.wx = torch.nn.Embedding(vocab_size, self.embed_size)
        self.cell = torch.nn.LSTM
        self.rnn = self.cell(input_size=self.embed_size,
                             hidden_size=self.embed_size,
                             batch_first=True)
        self.wy = torch.nn.Linear(in_features=self.embed_size,
                                  out_features=vocab_size)

        # init weights
        self.wx.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)
        max_w = np.sqrt(1 / self.embed_size)
        self.wy.weight.data.uniform_(-max_w, max_w)
        self.wy.bias.data.fill_(0.0)

        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden_state(self):
        hidden_state = torch.zeros(1, 1, self.embed_size).to(self.device)
        cell_state = torch.zeros(1, 1, self.embed_size).to(self.device)
        return hidden_state, cell_state

    def init_hidden_state_2d(self):
        hidden_state = torch.zeros(1, self.embed_size).to(self.device)
        cell_state = torch.zeros(1, self.embed_size).to(self.device)
        return hidden_state, cell_state

    def forward(self, token_id, h, c):
        """for each time step, predict the next token"""
        input_id = torch.tensor([token_id]).to(self.device)
        embeds = self.wx(input_id)
        outputs, (h, c) = self.rnn(embeds, (h, c))

        # get logits from output of rnn cell
        z_o = self.wy(outputs.reshape(-1, self.embed_size))
        o_prob = torch.nn.functional.softmax(z_o, dim=1)

        assert z_o.dim() == 2  # make sure first dimension is not removed when batch size = 1

        return z_o, o_prob, h, c
