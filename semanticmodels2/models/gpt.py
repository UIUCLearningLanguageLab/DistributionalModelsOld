from semanticmodels2.params import GPTParams, Params
from typing import List, Dict, Optional, Any
from collections import defaultdict, OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from semanticmodels2.datasets.corpus_xAyBz import Corpus_xAyBz
from pathlib import Path
import random


class GPT:
    def __init__(self,
                 params: GPTParams,
                 corpus: Corpus_xAyBz,
                 save_path: Optional[Path] = None):
        self.params = params
        self.corpus = corpus
        self.vocab_id_dict = corpus.vocab_id_dict
        self.id_vocab_dict = corpus.id_vocab_dict
        self.vocab_size = corpus.vocabulary_size
        self.numeric_document_list = copy.deepcopy(corpus.numeric_document_list)
        self.encode = lambda s: [self.vocab_id_dict[v] for v in s]
        self.decode = lambda l: [self.id_vocab_dict[i] for i in l]

        self.block_size = self.params.block_size
        self.batch_size = self.params.batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_interval = 10
        self.save_path = save_path
        self.performance = defaultdict(list)

        for document in self.numeric_document_list:
            print(f'Full document{document}')

        self.model = BigramLanguageModel(self.vocab_size,
                                         self.params.embed_size,
                                         self.block_size,
                                         self.params.head_size,
                                         self.params.num_heads,
                                         self.device)

        for contexts, targets in self.get_batch(self.numeric_document_list):
            logits, loss = self.model(contexts, targets)
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
            logits, loss = self.model(contexts, targets)
            total_loss += loss.item()
            num_loss += 1
        self.model.train()
        return total_loss / num_loss

    def train_epoch(self, numeric_document_list):
        self.model.train()
        for contexts, targets in self.get_batch(numeric_document_list):
            logits, loss = self.model(contexts, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.model.to(self.device)
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

            final_numeric_document_list = \
                [[token for sentence in doc for token in sentence] for doc in final_numeric_document_sentence_list]
            if epoch % self.eval_interval == 0:
                print(f"epoch {epoch}: train loss {self.evaluate(final_numeric_document_list):.4f}")
            self.train_epoch(final_numeric_document_list)
            if epoch <= 100 or (100 < epoch <= 500 and epoch % 5 == 0) or (
                    500 < epoch <= 1000 and epoch % 10 == 0) or (
                    1000 < epoch <= 10000 and epoch % 25 == 0) or (
                    10000 < epoch <= 20000 and epoch % 50 == 0):
                self.save_model(epoch, save_index)
                save_index += 1

    def test(self):
        self.model.eval()

    # save the model as pickel file
    def save_model(self, current_epoch, save_index):
        torch.save(self.model.state_dict(), self.save_path / '{}_{}_model.pt'.format(save_index, current_epoch))

    # getter function for performance dictionary
    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, head_size, num_head, device):
        super().__init__()
        # self.vocab_size = vocab_size
        # self.embed_size = embed_size
        # self.block_size = block_size
        self.device = device
        self.token_embeddings_table = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings_table = nn.Embedding(block_size, embed_size)
        self.sa_head = MultiHeadAttention(num_head, embed_size // num_head, embed_size, block_size)
        self.ffwd = FeedFoward(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets):
        B, T = idx.shape
        # idx and targets are both (B, T)
        token_embed = self.token_embeddings_table(idx)  # (Batch*Time*embed_size)
        position_embed = self.position_embeddings_table(torch.arange(T, device=self.device))
        x = token_embed + position_embed
        x = self.sa_head(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (Batch*Time*vocab_size)
        B, T, C = logits.shape
        print(T)
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss


class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B ,T, C)
        q = self.query(x)  # (B ,T, C)
        v = self.value(x)  # (B ,T, C)

        # wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention, original in paper "attention is all you neeed"
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B ,T, C) @  (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=1)  # (B, T, T)

        out = wei @ v  # (B, T, T) @ (B, T ,C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for i in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedFoward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, embed_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)
