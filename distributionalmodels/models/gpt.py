from distributionalmodels.params import GPTParams, Params
from typing import List, Dict, Optional, Any
from collections import defaultdict, OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from distributionalmodels.datasets.corpus_xAyBz import Corpus_xAyBz
from pathlib import Path
import random
import yaml
import os
import numpy as np


class GPT:
    # load w2v from saved pretrained model
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
            print(f'Loaded model from {model_file}')

        return dsm_list, checkpoint_list, corpus

    def __init__(self,
                 params: GPTParams,
                 corpus: Corpus_xAyBz,
                 save_path: Optional[Path] = None):
        self.params = params
        self.corpus = corpus
        self.vocab_id_dict = corpus.vocab_id_dict
        self.id_vocab_dict = corpus.id_vocab_dict
        self.encode = lambda s: [self.vocab_id_dict[v] for v in s]
        self.decode = lambda l: [self.id_vocab_dict[i] for i in l]
        self.vocab_size = corpus.vocabulary_size
        self.numeric_document_list = copy.deepcopy(corpus.numeric_document_list)

        self.block_size = self.params.block_size
        self.batch_size = self.params.batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_interval = 10
        self.save_path = save_path
        self.performance = defaultdict(list)
        #
        # for document in self.numeric_document_list:
        #     print(f'Full document{document[:10]}')

        self.model = BigramLanguageModel(self.vocab_size,
                                         self.params.embed_size,
                                         self.block_size,
                                         self.params.hidden_size,
                                         self.params.num_heads,
                                         self.device)
        self.model.to(device=self.device)

        for contexts, targets in self.get_batch(self.numeric_document_list):
            logits, loss, o_prob = self.model(contexts, targets)
            for b in range(self.batch_size):
                for t in range(self.block_size):
                    context = contexts[b, :t + 1]
                    target = targets[b, t]
                    # print(f'when input is {context.tolist()} the target: {target}')

    def get_batch(self, numeric_document_list):
        for document in numeric_document_list:
            data = torch.tensor(document, dtype=torch.long)
            all_index_list = list(range(len(document) - self.block_size))
            batches = [all_index_list[n:n + self.batch_size] for n in range(0, len(all_index_list), self.batch_size)]
            num_batches = len(batches)
            for batch_index in range(num_batches):
                x = torch.stack([data[i:i + self.block_size] for i in batches[batch_index]])
                y = torch.stack([data[i + 1:i + self.block_size + 1] for i in batches[batch_index]])
                x, y = x.to(self.device), y.to(self.device)
                yield x, y

    @torch.no_grad()
    def evaluate(self, numeric_document_list):
        self.model.eval()
        total_loss = 0
        num_loss = 0
        for contexts, targets in self.get_batch(numeric_document_list):
            logits, loss, o_prob = self.model(contexts, targets)
            total_loss += loss.item()
            num_loss += 1
        self.model.train()
        return total_loss / num_loss

    def train_epoch(self, numeric_document_list):
        self.model.train()
        for contexts, targets in self.get_batch(numeric_document_list):
            logits, loss, o_prob = self.model(contexts, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def test(self, numeric_document_list):
        self.model.eval()
        correctness = 0
        total_seq = 0
        for contexts, targets in self.get_batch(numeric_document_list):
            total_seq += 1
            B, T = contexts.shape
            logits, loss, o_prob = self.model(contexts, targets)
            predicted_tokens = torch.argmax(logits, dim=-1)
            input_sequences = contexts.view(B*T).tolist()
            predictions = predicted_tokens.tolist()
            target_sequences = targets.view(B*T).tolist()

            for i, prediction in enumerate(predictions):
                if prediction == target_sequences[i]:
                    correctness += 1

            print(self.decode(input_sequences))

            for input, prediction, target in zip(self.decode(input_sequences), self.decode(predictions),
                                                 self.decode(target_sequences)):
                print(f"Item: {input}, Predicted: {prediction}, Target: {target}")
        print(f'accuracy: {correctness/(total_seq * self.block_size)}')

    @torch.no_grad()
    def get_multihead_outputs(self, numeric_document_list):
        self.multihead_outputs = []
        self.attention_weights = []
        self.model.eval()
        multihead_forward_hook = self.model.sa_head.register_forward_hook(self.multihead_forward_hook)
        multihead_inputs = []
        o_probs = []
        for contexts, targets in self.get_batch(numeric_document_list):
            logits, loss, o_prob = self.model(contexts, targets)
            multihead_inputs.append(contexts)
            o_probs.append(o_prob)
        multihead_forward_hook.remove()
        return self.multihead_outputs, self.attention_weights, multihead_inputs, o_probs

    @torch.no_grad()
    def get_kqv_outputs(self, numeric_document_list):
        self.key_outputs = []
        self.query_outputs = []
        self.value_outputs = []
        self.model.eval()
        # Registering the hooks
        key_hook = self.model.sa_head.heads[0].key.register_forward_hook(self.key_hook)
        query_hook = self.model.sa_head.heads[0].query.register_forward_hook(self.query_hook)
        value_hook = self.model.sa_head.heads[0].value.register_forward_hook(self.value_hook)
        inputs = []
        for contexts, targets in self.get_batch(numeric_document_list):
            logits, loss, o_prob = self.model(contexts, targets)
            inputs.append(contexts)

        key_hook.remove()
        query_hook.remove()
        value_hook.remove()
        return self.key_outputs, self.query_outputs, self.value_outputs, inputs

    def train(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.learning_rate)
        save_index = 0
        for epoch in range(self.params.num_epochs):
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

            # += instead of append
            final_numeric_document_list = \
                [[token for sentence in doc for token in sentence] for doc in final_numeric_document_sentence_list]
            if epoch % self.eval_interval == 0:
                print(f"epoch {epoch}: train loss {self.evaluate(final_numeric_document_list):.4f}")
            self.train_epoch(final_numeric_document_list)
            # if epoch <= 100 or (100 < epoch <= 500 and epoch % 5 == 0) or (
            #         500 < epoch <= 1000 and epoch % 10 == 0) or (
            #         1000 < epoch <= 10000 and epoch % 25 == 0) or (
            #         10000 < epoch <= 20000 and epoch % 50 == 0):
            if epoch % 1000 == 0:
                self.save_model(epoch, save_index)
                save_index += 1
        self.test(self.numeric_document_list)

    # save the model as pickel file
    def save_model(self, current_epoch, save_index):
        torch.save(self.model.state_dict(), self.save_path / '{}_{}_model.pt'.format(save_index, current_epoch))

    # getter function for performance dictionary
    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    # register forward hook
    def multihead_forward_hook(self, module, input, output):
        combined_output, attention_weight = output
        self.multihead_outputs.append(combined_output)
        self.attention_weights.append(attention_weight)

    def key_hook(self, module, input, output):
        self.key_outputs.append(output)

    def query_hook(self, module, input, output):
        self.query_outputs.append(output)

    def value_hook(self, module, input, output):
        self.value_outputs.append(output)



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, hidden_size, num_head, device):
        super().__init__()
        # self.vocab_size = vocab_size
        # self.embed_size = embed_size
        # self.block_size = block_size
        self.device = device
        self.token_embeddings_table = nn.Embedding(vocab_size, embed_size).to(self.device)
        self.position_embeddings_table = nn.Embedding(block_size, embed_size).to(self.device)
        self.sa_head = MultiHeadAttention(num_head, embed_size // num_head, embed_size, block_size)
        self.ffwd = FeedForward(embed_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape
        # idx and targets are both (B, T)
        token_embed = self.token_embeddings_table(idx)  # (Batch*Time*embed_size)
        position_embed = self.position_embeddings_table(torch.arange(T, device=self.device))
        x = token_embed + position_embed
        x, attention_weights = self.sa_head(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (Batch*Time*vocab_size)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)
        o_prob = torch.nn.functional.softmax(logits, dim=1)
        return logits, loss, o_prob


class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_mask_attention_weights = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B ,T, C)
        q = self.query(x)  # (B ,T, C)
        v = self.value(x)  # (B ,T, C)
        # wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention, original in paper "attention is all you need"
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
        self.pre_mask_attention_weights = wei.clone()
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=1)  # (B, T, T)

        out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for i in range(num_heads)])

    def forward(self, x):
        attention_weights_list = []
        head_outputs = []
        for head in self.heads:
            head_output = head(x)
            head_outputs.append(head_output)

            # Assuming pre_mask_attention_weights are stored in each head
            attention_weights_list.append(head.pre_mask_attention_weights)

        # Concatenating the outputs from all heads
        combined_output = torch.cat(head_outputs, dim=-1)

        # Average the attention weights across all heads
        combined_attention_weights = torch.cat(attention_weights_list, dim=-1)
        return combined_output, combined_attention_weights


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, hidden_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)
