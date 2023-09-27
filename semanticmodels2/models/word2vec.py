import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch import nn
from semanticmodels2.params import Params
from semanticmodels2.datasets.corpus_xAyBz import Corpus_xAyBz
import torch
import copy
import logging
from gensim.models import Word2Vec
from torch.optim.lr_scheduler import LambdaLR
from semanticmodels2.params import Word2VecParams
from pathlib import Path
import yaml
import random
import os

random.seed(1023)
class W2Vec:
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
        model_files = sorted(model_files,key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))
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
            # print(f'Loaded model from {model_file}'

        return dsm_list, checkpoint_list, corpus


    def __init__(self,
                 params: Word2VecParams,
                 corpus: Corpus_xAyBz,
                 save_path: Optional[Path] = None,
                 ):
        
        self.params = params
        self.corpus = corpus
        self.save_path = save_path
        self.vocab_size = corpus.vocabulary_size
        self.vocab_id_dict = corpus.vocab_id_dict
        self.id_vocab_dict = corpus.id_vocab_dict

        self.numeric_document_list = copy.deepcopy(corpus.numeric_document_list)

        self.window_size = self.params.window_size
        self.dataset = None
        
        self.model = TorchWord2Vec(self.vocab_size, self.params.embed_size, self.params.embed_init_range)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
        self.performance = defaultdict(list)

    def windowizer(self, numeric_document_list):
        """
        Windowizer function for Word2Vec. Converts sentence to sliding-window
        pairs.
        """
        docs = numeric_document_list
        wsize = self.window_size
        out = []
        for doc in docs:
            for i, wd in enumerate(doc):
                target = wd
                window = [i+j for j in
                        range(-wsize, wsize+1, 1)
                        if (i+j>=0) &
                            (i+j<len(doc)) &
                            (j!=0)]
                out+=[(target, doc[w]) for w in window]
        return out
    
    def gen_batches(self, target_context_pairs, BATCH_SIZE=1, shuffle=False):
        dataloader = DataLoader(Word2VecDataset(target_context_pairs, vocab_size=self.vocab_size),
                                                batch_size=BATCH_SIZE,
                                                shuffle=shuffle)
        return dataloader

    def calc_pp(self,
                target_context_pairs,  # sequences of token IDs
                shuffle: bool,
                ):
        self.model.eval()
        num_loss = 0
        loss_total = 0
        for i, data in enumerate(self.gen_batches(target_context_pairs, shuffle=shuffle), 0):
            center, context = data
            center, context = center.to(self.device), context.to(self.device)
            self.optimizer.zero_grad()
            logits, hidden = self.model(input=center)
            loss = self.criterion(logits, context)
            loss_total += loss.item()
            num_loss += 1

        cross_entropy_average = loss_total / num_loss
        res = np.exp(cross_entropy_average)  # converting cross-entropy loss to perplexity score
        return res

    def train_epoch(self, target_context_pairs, shuffle:bool):
        self.model.train()
        for i, data in enumerate(self.gen_batches(target_context_pairs, shuffle=shuffle), 0):
            center, context = data
            center, context = center.to(self.device), context.to(self.device)
            self.optimizer.zero_grad()
            logits, hidden = self.model(input=center)
            loss = self.criterion(logits, context)
            loss.backward()
            self.optimizer.step()
    
    
    def train(self,
              output_freq: int = 2):

        self.model.to(self.device)  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()  # loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.params.num_epochs, verbose=False)
        
        save_index = 1   
        train_target_context_pairs = self.windowizer(self.numeric_document_list)
        pp_train = self.calc_pp(train_target_context_pairs, shuffle=False)
        self.performance['epoch'].append(0)
        self.performance['pp_train'].append(pp_train)
        print(f'0: {pp_train:8.2f}')

        for epoch in range(1, self.params.num_epochs + 1):
            # train on one epoch
            self.train_epoch(train_target_context_pairs, shuffle=True)

            self.lr_scheduler.step()

            if epoch % output_freq == 0:
                self.performance['epoch'].append(epoch)
                pp_train = self.calc_pp(train_target_context_pairs, shuffle=False)
                self.performance['pp_train'].append(pp_train)
                print(f'{epoch}: {pp_train:8.2f}')
                save_index+=1
                self.save_model(epoch, save_index)

        pp_train = self.calc_pp(train_target_context_pairs, shuffle=False)
        self.performance['pp_train'].append(pp_train)
        self.performance['epoch'].append(self.params.num_epochs +1)
        self.save_model(epoch, save_index+1)
        print(f'Train perplexity after training: {pp_train:8.2f}')

    def genism_train(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        sg = Word2Vec(sentences=self.seq_tok[0],
                      sg=True,
                      vector_size=self.params.embed_size,
                      window=4,
                      epochs=self.params.num_epochs,
                      alpha=self.params.learning_rate,  # originally 0.025
                      min_count=1,
                      hs=1,  # better accuracy when hs=1
                      )
        self.t2e = {t: np.asarray(sg.wv[t]) for t in self.vocab}

    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    def save_model(self, current_epoch, save_index):
        torch.save(self.model.state_dict(), self.save_path / '{}_{}_model.pt'.format(save_index, current_epoch))

class TorchWord2Vec(nn.Module):
    def __init__(self, vocab_size,
                 embedding_size,
                 embed_init_range):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embed_init_range = embed_init_range

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.wy = nn.Linear(embedding_size, vocab_size)
        self.sigmoid = nn.Sigmoid().float()

        nn.init.uniform_(self.embed.weight, -self.embed_init_range, self.embed_init_range)
        nn.init.uniform_(self.wy.weight, -self.embed_init_range, self.embed_init_range)

    def forward(self, input):
        # Encode input to lower-dimensional representation
        hidden = self.embed(input)
        # Expand hidden layer to predictions
        logits = self.wy(hidden)
        assert logits.dim() == 2
        return logits, hidden

class Word2VecDataset(Dataset):

    def __init__(self, dataset, vocab_size):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return lr_scheduler