import random
from semanticmodels2.datasets.dataset import Dataset

class Test(Dataset):

    def __init__(self, num_documents, sent_per_doc):
        super().__init__()


        self.document_list = None
        self.num_documents = num_documents
        self.sentences_per_document = sent_per_doc

        self.category_dict = {'DET': ['the', 'a'],
                              'CARNIVORE': ['bear', 'cheetah', 'lion', 'tiger'],
                              'HERBIVORE': ['deer', 'moose', 'zebra', 'elephant'],
                              'PLANT': ['tree', 'grass', 'bush', 'flower'],
                            
                              'ANIMAL_TVERB': ['watched', 'ate'], 
                              'CARNIVORE_TVERB': ['chased', 'attacked'],
                              'HERBIVORE_TVERB': ['escaped', 'hid_from'],
                              'ANIMAL_IVERB': ['slept', 'sat', 'walked', 'turned'],
                              'LIVING_IVERB': ['grew', 'died', 'fell', 'aged'],
                              'PUNCT': ['.']}
        
        self.create_documents()


    def create_token_data(self):
        pass

    def create_type_data(self):
        pass

    def create_numeric_token_sequence_list(self):
        pass

    def print_dataset(self):
        print(self.document_list)

    def create_documents(self):
        self.document_list = []
        for i in range(self.num_documents):
            current_document = []
            for j in range(self.sentences_per_document):
                sentence = self.create_sentence()
                self.document_list += sentence
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
                if verb == 'watched':
                    theme_category = random.choice(['HERBIVORE', 'CARNIVORE', 'PLANT'])
                    theme = random.choice(self.category_dict[theme_category])
                    sentence_list.append(theme)
                elif verb == 'ate':
                    theme = random.choice(self.category_dict['PLANT'])
                    sentence_list.append(theme)
            if verb_category == 'HERBIVORE_TVERB':
                verb = random.choice(self.category_dict['HERBIVORE_TVERB'])
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
                if verb == 'watched':
                    theme_category = random.choice(['HERBIVORE', 'CARNIVORE', 'PLANT'])
                    theme = random.choice(self.category_dict[theme_category])
                    sentence_list.append(theme)
                elif verb == 'ate':
                    theme = random.choice(self.category_dict['HERBIVORE'])
                    sentence_list.append(theme)
            if verb_category == 'CARNIVORE_TVERB':
                verb = random.choice(self.category_dict['CARNIVORE_TVERB'])
                theme = random.choice(self.category_dict['HERBIVORE'])
                sentence_list.append(theme)

        punct = random.choice(self.category_dict['PUNCT'])
        sentence_list.append(punct)

def main():
    num_documents = 10
    sent_per_doc = 5
    test_dataset = Test(num_documents, sent_per_doc)
    test_dataset.print_dataset()

main()
