import torch
import numpy as np
import sys
from typing import List, Dict, Optional, Any
from collections import defaultdict
import pandas as pd
from pathlib import Path
import torch
import copy
import pyprind
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class RNN:

    def __init__(self,
                 params: Dict[str, Any],  # not everything is a float
                 vocab_dict: Dict[str, int],
                 numeric_sequence_list: List[List[int]],  # sequences of token IDs, "numeric sequences"
                 # exp2b specific arguments:
                 df_blank: Optional[pd.DataFrame] = None,
                 instruments: Optional[List[str]] = None,
                 save_path: Optional[Path] = None,
                 ):
        self.params = params
        self.vocab_dict = vocab_dict
        self.numeric_sequence_list = copy.deepcopy(numeric_sequence_list)
        self.df_blank = df_blank
        self.instruments = instruments
        self.save_path = save_path

        self.vocab_size = len(vocab_dict)
        self.id2token = {i: token for token, i in self.vocab_dict.items()}

        self.model = TorchRNN(self.params['rnn_type'],
                              self.params['num_layers'],
                              self.params['embed_size'],
                              self.params['embed_init_range'],
                              self.params['dropout_prob'],
                              self.vocab_size,
                              self.vocab_dict,
                              )

        self.t2e = None
        self.performance = defaultdict(list)

    def gen_batches(self,
                    numeric_sequence_list: List[List[int]],  # sequences of token IDs, e.g. [[0, 5, 334,], [,]  ]
                    batch_size: Optional[int] = None,
                    group_by_lenth: bool =  False,
                    padding: bool = False,
                    ):
        """
        generate sequences for predicting next-tokens.
        Note:
        each token in each sequence must be predicted during training.
        this function does not return moving windows.
        """

        print(f'Generating batches for {len(numeric_sequence_list)} sequences', flush=True)

        if batch_size is None:
            batch_size = self.params['batch_size']

        # shuffle and flatten
        np.random.shuffle(numeric_sequence_list)

        # separate sequences by length to avoid padding batches
        if group_by_lenth:
            length2seq_group = defaultdict(list)
            for s in numeric_sequence_list:
                length2seq_group[len(s)].append(s)
            for k, v in length2seq_group.items():
                print(f'Found {len(v):>12,} sequences with length={k:>6}', flush=True)

        # collect sequences of same length into batches
            for seq_len, seq_group in length2seq_group.items():

                is_leftover_batch = len(seq_group) % batch_size != 0
                num_batches = len(seq_group) // batch_size + int(is_leftover_batch)
                for i in range(num_batches):
                    start = batch_size * i
                    seq_b = seq_group[start:start + batch_size]  # if end index is too large, small batch is created
                    if len(seq_b) != batch_size:
                        print(f'Found abnormal batch_size={len(seq_b)}')  # this is by design

                    yield seq_b  # this is the batch used for training
        else:
            is_leftover_batch = len(numeric_sequence_list) % batch_size != 0
            num_batches = len(numeric_sequence_list) // batch_size + int(is_leftover_batch)
            for i in range(num_batches):
                start = batch_size * i
                seq_b = numeric_sequence_list[start:start + batch_size]  # if end index is too large, small batch is created
                if len(seq_b) != batch_size:
                    print(f'Found abnormal batch_size={len(seq_b)}')  # this is by design

                yield seq_b  # this is the batch used for training

    def calc_pp(self,
                seq_num: List[List[int]],  # sequences of token IDs
                verbose: bool,
                ):
        if verbose:
            print('Calculating perplexity...')

        self.model.eval()

        loss_total = 0
        num_loss = 0
        for seq_b in self.gen_batches(seq_num):

            # forward step
            logits = self.model(seq_b)
            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            list_of_label_tensors = [torch.LongTensor(seq).to(torch.device("mps"))[1:] for seq in seq_b]  # get all time steps excluding the first
            labels = pad_sequence(list_of_label_tensors, batch_first=True, padding_value=self.vocab_dict['PAD'])
            # labels = torch.LongTensor(seq_b).to(torch.device('mps'))[:, 1:]
            labels = torch.flatten(labels)
            # logits and labels do not include empty/padded time steps
            loss = self.criterion(logits,  # [batch_size * seq_len, vocab_size]
                                  labels,  # [batch_size * seq_len]
                                  # ignore_index=self.vocab_dict['PAD'],  # probably don't need to do that
                                  )
            loss_total += loss.item()
            num_loss += 1

        cross_entropy_average = loss_total / num_loss
        res = np.exp(cross_entropy_average)  # converting cross-entropy loss to perplexity score
        return res
    def train_item(self, input, old_hidden, output):
        hidden, outputs = self.model(input)
        self.optimizer.zero_grad()


    def train_epoch(self,
                    numeric_sequence_list: List[List[int]],  # sequences of token IDs
                    ) -> None:
        self.model.train()

        num_batches = 0
        for seq_b in self.gen_batches(numeric_sequence_list):  # generates batches of complete sequences
            # forward step
            # do not include last time step in input
            logits = self.model(seq_b)

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            list_of_label_tensors = [torch.LongTensor(seq).to(torch.device("mps"))[1:] for seq in seq_b]  # get all time steps excluding the first
            labels = pad_sequence(list_of_label_tensors, batch_first=True, padding_value=self.vocab_dict['PAD'])
            # labels = torch.LongTensor(seq_b).to(torch.device('mps'))[:, 1:]
            labels = torch.flatten(labels)

            # logits and labels do not include empty/padded time steps
            loss = self.criterion(logits,  # [batch_size * seq_len, vocab_size]
                                  labels,  # [batch_size * seq_len]
                                  # ignore_index=self.vocab_dict['PAD'],  # probably don't need to do that
                                  )
            loss.backward()

            # gradient clipping + update weights
            if self.params['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.params['grad_clip'],
                                               norm_type=2)
            self.optimizer.step()  # weights are updated

            num_batches += 1

        print(f'Trained on {num_batches} batches (={num_batches * self.params["batch_size"]} sequences)', flush=True)

    def train(self,
              verbose: bool = True,
              calc_pp_train_during_training: bool = True,
              calc_pp_train_after_training: bool = False,
              save_inferences_during_training: bool = False,
              ):

        self.model.to(torch.device("mps"))  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()  # loss function
        self.optimizer = torch.optim.Adagrad(self.model.parameters(),
                                             lr=self.params['learning_rate'],
                                             lr_decay=self.params['lr_decay'],
                                             weight_decay=self.params['weight_decay'])
        # split data
        train_numeric_sequence_list = []
        valid_numeric_sequence_list = []
        test_numeric_sequence_list = []
        for seq_num_i in self.numeric_sequence_list:
            if np.random.binomial(1, self.params['train_percent']):
                train_numeric_sequence_list.append(seq_num_i)
            else:
                if np.random.binomial(1, 0.5):  # split valid and test docs evenly
                    valid_numeric_sequence_list.append(seq_num_i)
                else:
                    test_numeric_sequence_list.append(seq_num_i)
        print(f'Num sequences in train={len(train_numeric_sequence_list):,}')
        print(f'Num sequences in valid={len(valid_numeric_sequence_list):,}')
        print(f'Num sequences in test ={len(test_numeric_sequence_list):,}')

        # get unique sequences in train data for evaluating train_pp

        train_numeric_sequence_list_unique = []
        pbar = pyprind.ProgBar(len(train_numeric_sequence_list), stream=sys.stdout)
        for s in train_numeric_sequence_list:
            # if s not in train_numeric_sequence_list_unique:
            train_numeric_sequence_list_unique.append(s)
            pbar.update()
        print(f'Num unique sequences in train ={len(train_numeric_sequence_list_unique):,}')

        if calc_pp_train_during_training:
            pp_train = self.calc_pp(train_numeric_sequence_list_unique, verbose)
            self.performance['epoch'].append(0)
            self.performance['pp_train'].append(pp_train)
            print(f'Train perplexity at epoch {0}: {pp_train:8.2f}')

        # save during-training results to disk (for plotting learning curves)
        if save_inferences_during_training:
            self.fill_in_blank_df_and_save(0)  # specific to Shufan's project

        # train loop
        pbar = pyprind.ProgBar(self.params['num_epochs'], stream=sys.stdout)
        for epoch in range(1, self.params['num_epochs'] + 1):
            self.performance['epoch'].append(epoch)

            if verbose:
                print()
                print(f'Epoch {epoch:>6}', flush=True)

            # train on one epoch
            self.train_epoch(train_numeric_sequence_list)

            # save during-training results to disk (for plotting learning curves)
            if save_inferences_during_training:
                self.fill_in_blank_df_and_save(epoch)

            if self.params['train_percent'] < 1.0:
                pp_val = self.calc_pp(valid_numeric_sequence_list, verbose)
                self.performance['pp_val'].append(pp_val)
                if verbose:
                    print(f'Validation perplexity at epoch {epoch}: {pp_val:8.2f}')

            if calc_pp_train_during_training:
                pp_train = self.calc_pp(train_numeric_sequence_list_unique, verbose)
                self.performance['pp_train'].append(pp_train)
                print(f'Train perplexity at epoch {epoch}: {pp_train:8.2f}')

            if not verbose:
                pbar.update()

        if self.params['train_percent'] < 1.0:
            pp_val = self.calc_pp(valid_numeric_sequence_list, verbose)
            if verbose:
                print(f'Validation perplexity after training: {pp_val:8.2f}')

        if calc_pp_train_after_training:
            pp_train = self.calc_pp(train_numeric_sequence_list, verbose)
            self.performance['pp_train'].append(pp_train)
            self.performance['epoch'].append(self.performance['epoch'][-1] + 1)
            print(f'Train perplexity after training: {pp_train:8.2f}')

    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    def calc_native_sr_scores(self,  # this is specific Shufan's project
                              verb: str,
                              theme: str,
                              instruments: List[str],
                              verbose: bool = True,
                              ) -> List[float]:
        """
        use language modeling based prediction task to calculate "native" sr scores
        """

        # TODO does Agent need to be in input to perform well on exp2b?

        # prepare input
        token_ids = [self.token2id['John'], self.token2id[verb], self.token2id[theme]]
        if 'with' in self.token2id:
            token_ids.append(self.token2id['with'])

        # get logits (at last time step)
        with torch.no_grad():
            x_b = [token_ids]
            logits_at_last_step = self.model.predict_next_token(torch.LongTensor(x_b).device("mps"))  # [1, vocab_size]
            logits_at_last_step = logits_at_last_step.squeeze()  # [vocab_size]

        # these are printed to console
        exp_vps = {'preserve pepper',
                   # 'preserve orange',
                   # 'repair blender',
                   # 'repair bowl',
                   # 'pour tomato-juice',
                   # 'decorate cookie',
                   # 'carve turkey',
                   # 'heat tilapia',
                   }
        vps = {'grow potato',
               }

        # get scores
        scores = []
        instrument2sr = {}
        for instrument in instruments:
            token_id = self.token2id[instrument]
            sr = logits_at_last_step[token_id].item()
            scores.append(sr)
            instrument2sr[instrument] = sr

        for instrument, sr in sorted(instrument2sr.items(), key=lambda i: i[1]):
            if verbose and verb + ' ' + theme in vps:
                print(f'{verb} {theme} {instrument:>12} : {sr: .4f}')
        if verbose and verb + ' ' + theme in vps:
            print()

        return scores

    def fill_in_blank_df_and_save(self, epoch: int):  # specific to Shufan's project
        """
        fill in blank data frame with semantic-relatedness scores
        """
        self.model.eval()

        if (self.df_blank is None) or (self.instruments is None) or (self.save_path is None):
            raise RuntimeError('To fill in blank sr dataframe,'
                               ' RNN must be provided with blank df, instruments, and save path.')

        df_results = self.df_blank.copy()

        for verb_phrase, row in self.df_blank.iterrows():
            verb_phrase: str
            verb, theme = verb_phrase.split()
            scores = self.calc_native_sr_scores(verb, theme, self.instruments)
            df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type'], row['location-type']] + scores

        df_results.to_csv(self.save_path / f'df_sr_{epoch:06}.csv')


class TorchRNN(torch.nn.Module):
    def __init__(self,
                 rnn_type: str,
                 num_layers: int,
                 embed_size: int,
                 embed_init_range: float,
                 dropout_prob: float,
                 vocab_size: int,
                 vocab_dict: dict,
                 ):
        super().__init__()
        self.rnn_type = rnn_type
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.vocab_dict = vocab_dict

        self.wx = torch.nn.Embedding(vocab_size, self.embed_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=embed_size,
                             hidden_size=embed_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout_prob)
        self.wy = torch.nn.Linear(in_features=embed_size,
                                  out_features=vocab_size)

        # init weights
        self.wx.weight.data.uniform_(-embed_init_range, embed_init_range)
        max_w = np.sqrt(1 / self.embed_size)
        self.wy.weight.data.uniform_(-max_w, max_w)
        self.wy.bias.data.fill_(0.0)

    def predict_next_token(self, input_ids):
        """predict only the next token only for the last time step"""
        embeds = self.wx(input_ids)
        outputs, hidden = self.rnn(embeds)  # this returns all time steps
        
      # outputs has shape (batch size, sequence length, hidden size)
        final_outputs = torch.squeeze(outputs[:, -1])
        logits = self.wy(final_outputs)

        # keep first dim
        if len(input_ids) == 1:
            logits = torch.unsqueeze(logits, 0)

        return logits

    def forward(self,
                input,
                use_packing: bool = True,
                forward_item: bool = True
                ):
        """for each time step, predict the next token"""
        
        if forward_item:
            input_id = torch.LongTensor(input)
            embeds = self.wx(input_id)
            outputs, hidden = self.rnn(embeds, hidden)

        if use_packing:
            input_ids = [torch.LongTensor(seq).to(torch.device("mps"))[:-1] for seq in input]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.vocab_dict['PAD'])
            seq_len = [len(vector) for vector in input_ids]
            embeds = self.wx(input_ids_padded)
            # packed_input = pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
            packed_outputs, hidden = self.rnn(embeds,hidden)
            # packed outputs is not a tensor, it is a list of sequences
            # we need to convert it to a tensor of shape [batch size * ( seq len 1 + seq len 2 + ...), embed_size]
            # print(packed_outputs.data.shape)
            # for tensor in packed_outputs.data:
            #     assert not torch.all(torch.isclose(tensor.cpu(), torch.zeros_like(tensor).cpu()))
            # outputs = packed_outputs.data

            # i don't think we need this
            # outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0)
            outputs = packed_outputs

        else:
            input_ids = torch.LongTensor(input).to(torch.device('mps'))[:, :-1]
            embeds = self.wx(input_ids)
            outputs, hidden = self.rnn(embeds)  # this returns all time steps

        # get logits from output of rnn cell
        logits = self.wy(outputs.reshape(-1, self.embed_size))

        assert logits.dim() == 2  # make sure first dimension is not removed when batch size = 1

        return hidden, logits
