import random
from distributionalmodels.datasets.dataset import Dataset
from functools import reduce 
from operator import iconcat
from typing import Dict, List
import csv
import time

class TestDataset(Dataset):

    def __init__(self, corpus_name, num_documents, sent_per_doc):
        super().__init__()

        self.corpus_name = corpus_name
        self.document_list = None
        self.sentence_list = None
        self.num_documents = num_documents
        self.sentences_per_document = sent_per_doc

        self.category_dict = {'DET': ['the', 'a'],
                              'CARNIVORE': ['bear', 'cheetah', 'lion', 'tiger'],
                              'HERBIVORE': ['deer', 'moose', 'zebra', 'elephant'],
                              'PLANT': ['tree', 'vine', 'bush', 'flower'],
                            
                              'ANIMAL_TVERB': ['watched', 'ate'], 
                              'CARNIVORE_TVERB': ['chased', 'attacked'],
                              'HERBIVORE_TVERB': ['escaped', 'hid_from'],
                              'ANIMAL_IVERB': ['slept', 'sat', 'walked', 'turned'],
                              'LIVING_IVERB': ['grew', 'died', 'fell', 'aged'],
                              'PUNCT': ['.']}
        
        start_time = time.time()
        self.create_documents()
        self.create_vocab_data()
        self.create_category_csv()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time for creating test_corpus is :', elapsed_time, 'seconds')

    def __repr__(self):
        output_string = "\nTest Dataset: {}\n".format(self.corpus_name)
        output_string += "\n"
        return output_string


    def create_vocab_data(self):
        self.token_list = reduce(iconcat, self.document_list, [])
        self.type_list = sorted(set(self.token_list))
        self.vocab_dict = Dict[str, int]
        self.vocab_dict = {token: token_id + 1 for token_id, token in enumerate(self.type_list)}
        self.vocab_dict['PAD'] = 0

        self.numeric_token_sequence_list: List[List[int]] = []
        self.numeric_token_sequence_sentence_list: List[List[int]] = []
        for document in self.document_list:
            document_sequence_list = [self.vocab_dict[token] for token in document]
            self.numeric_token_sequence_list.append(document_sequence_list)
        for sentence in self.sentence_list:
            sentence_sequence_list = [self.vocab_dict[token] for token in sentence]
            self.numeric_token_sequence_sentence_list.append(sentence_sequence_list)
        # print(self.numeric_token_sequence_list)

    def create_documents(self):
        self.document_list = []
        self.sentence_list = []
        for i in range(self.num_documents):
            current_document = []
            for j in range(self.sentences_per_document):
                sentence = self.create_sentence()
                self.sentence_list.append(sentence)
                current_document += sentence
            self.document_list.append(current_document)

    def create_sentence(self):
        sentence_list = []

        determiner = random.choice(self.category_dict['DET'])
        sentence_list.append(determiner)

        subject_category = random.choice(['PLANT', 'CARNIVORE', 'HERBIVORE'])
        subject = random.choice(self.category_dict[subject_category])
        sentence_list.append(subject)

        if subject_category == 'PLANT':
            verb = random.choice(self.category_dict['LIVING_IVERB'])
            sentence_list.append(verb)

        if subject_category == 'HERBIVORE':
            verb_category = random.choice(['LIVING_IVERB', 'ANIMAL_IVERB', "ANIMAL_TVERB", "HERBIVORE_TVERB"])
            if verb_category == 'LIVING_IVERB':
                verb = random.choice(self.category_dict['LIVING_IVERB'])
                sentence_list.append(verb)
            if verb_category == 'ANIMAL_IVERB':
                verb = random.choice(self.category_dict['ANIMAL_IVERB'])
                sentence_list.append(verb)
            if verb_category == 'ANIMAL_TVERB':
                verb = random.choice(self.category_dict['ANIMAL_TVERB'])
                sentence_list.append(verb)
                determiner = random.choice(self.category_dict['DET'])
                sentence_list.append(determiner)
                if verb == 'watched':
                    theme_category = random.choice(['HERBIVORE', 'CARNIVORE', 'PLANT'])
                    theme = random.choice(self.category_dict[theme_category])
                    sentence_list.append(theme)
                elif verb == 'ate':
                    theme = random.choice(self.category_dict['PLANT'])
                    sentence_list.append(theme)
                    
            if verb_category == 'HERBIVORE_TVERB':
                verb = random.choice(self.category_dict['HERBIVORE_TVERB'])
                sentence_list.append(verb)
                determiner = random.choice(self.category_dict['DET'])
                sentence_list.append(determiner)
                theme = random.choice(self.category_dict['CARNIVORE'])
                sentence_list.append(theme)

        if subject_category == 'CARNIVORE':
            verb_category = random.choice(['LIVING_IVERB','ANIMAL_IVERB', "ANIMAL_TVERB", "CARNIVORE_TVERB"])
            if verb_category == 'LIVING_IVERB':
                verb = random.choice(self.category_dict['LIVING_IVERB'])
                sentence_list.append(verb)
            if verb_category == 'ANIMAL_IVERB':
                verb = random.choice(self.category_dict['ANIMAL_IVERB'])
                sentence_list.append(verb)
            if verb_category == 'ANIMAL_TVERB':
                verb = random.choice(self.category_dict['ANIMAL_TVERB'])
                sentence_list.append(verb)
                determiner = random.choice(self.category_dict['DET'])
                sentence_list.append(determiner)
                if verb == 'watched':
                    theme_category = random.choice(['HERBIVORE', 'CARNIVORE', 'PLANT'])
                    theme = random.choice(self.category_dict[theme_category])
                    sentence_list.append(theme)
                elif verb == 'ate':
                    theme = random.choice(self.category_dict['HERBIVORE'])
                    sentence_list.append(theme)
            if verb_category == 'CARNIVORE_TVERB':
                verb = random.choice(self.category_dict['CARNIVORE_TVERB'])
                sentence_list.append(verb)
                determiner = random.choice(self.category_dict['DET'])
                sentence_list.append(determiner)
                theme = random.choice(self.category_dict['HERBIVORE'])
                sentence_list.append(theme)

        punct = random.choice(self.category_dict['PUNCT'])
        sentence_list.append(punct)
        return sentence_list

    def create_category_csv(self):
        f = open("datasets/test_dataset_categories.csv", 'w')
        for category in self.category_dict:
            word_list = self.category_dict[category]
            for word in word_list:
                f.write("{},{}\n".format(category, word))
        f.close()








