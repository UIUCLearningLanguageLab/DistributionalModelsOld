import os
import re
import spacy
import pickle
from collections import Counter
from distributionalmodels.datasets.dataset import Dataset

class Corpus(Dataset):
    """
    A class to represent a corpus of documents.
    """

    def __init__(self):
        super().__init__()
        """
        Initializes an empty corpus.
        """
        # Lists to store document titles and their corresponding content
        self.document_title_list = None
        self.document_list = None
        self.numeric_document_list = None
        self.num_documents = None

        # Configuration parameters for corpus creation
        self.input_path = None
        self.get_spacy_tokens = None
        self.create_char_corpus = None

        # Word-related attributes
        self.word_list = None
        self.num_words = None
        self.word_freq_dict = None
        self.word_token_list = None
        self.num_word_tokens = None
        self.word_pos_freq_dict = None
        self.vocab_id_dict = None
        self.id_vocab_dict = None
        self.vocabulary_size = None # make sure it is something get passed from params.py

        # Character-related attributes
        self.character_list = None
        self.num_characters = None
        self.character_index_dict = None
        self.character_freq_dict = None
        self.character_token_list = None
        self.num_character_tokens = None

    def __repr__(self):
        """
        Returns a string representation of the corpus.
        """
        return "CorpusObject: {} docs, {}/{} tokens, {}/{} characters".format(
            self.num_documents, self.num_words, self.num_word_tokens, 
            self.num_characters, self.num_character_tokens)

    def create_corpus(self, input_path, get_spacy_tokens=False, num_docs=None, create_char_corpus=False):
        """
        Creates a corpus from the documents in the specified directory.
        """
        self.input_path = input_path
        self.num_documents = num_docs
        self.create_char_corpus = create_char_corpus
        self.get_spacy_tokens = get_spacy_tokens
        self.get_numeric_document_lists()
        self.process_numeric_document_list()
        # print(self)

    def process_numeric_document_list(self):
        """
        Processes the loaded documents to generate word and character tokens.
        """
        # Initialize word-related attributes
        self.num_word_tokens = 0
        self.word_token_list = []
        self.word_freq_dict = Counter()
        self.word_pos_freq_dict = {}

        # Initialize character-related attributes
        self.num_character_tokens = 0
        self.character_freq_dict = Counter()
        self.character_token_list = []
        self.numeric_document_list = []

        # load spacy language model if we're going to use it
        if self.get_spacy_tokens:
            nlp = spacy.load("en_core_web_lg")

        for i in range(self.num_documents):
            print(f"Loading {self.document_title_list[i]}   {i}/{self.num_documents}")

            document_string = self.document_list[i]
            preprocessed_document_string = self.preprocess_document(document_string)

            # Process characters if required
            if self.create_char_corpus:
                character_list = list(preprocessed_document_string)
                self.character_freq_dict += Counter(character_list)
                self.character_token_list.append(character_list)
                self.num_character_tokens += len(character_list)

            # Process words using Spacy if required
            if self.get_spacy_tokens:
                doc = nlp(preprocessed_document_string)
                word_tokens = [token.text for token in doc]
                pos_list = [token.pos_ for token in doc]
                self.num_word_tokens += len(word_tokens)

                for j in range(len(word_tokens)):
                    word = word_tokens[j]
                    pos = pos_list[j]
                    if pos not in self.word_pos_freq_dict:
                        self.word_pos_freq_dict[pos] = {}
                    if word not in self.word_pos_freq_dict[pos]:
                        self.word_pos_freq_dict[pos][word] = 0
                    self.word_pos_freq_dict[pos][word] += 1
            else:
                # Process words without Spacy
                punct_dict = {".": "PERIOD", "!": "EXCLAIM", "?": "QUESTION", ",": "COMMA", ";": "COLON", ":": "SEMICOLON"}
                preprocessed_document_string = preprocessed_document_string.lower()
                preprocessed_document_string = preprocessed_document_string.replace("+", "_")
                preprocessed_document_string = preprocessed_document_string.replace("-", "_")
                start_tokens = preprocessed_document_string.split()

                word_tokens = []
                for token in start_tokens:
                    if token[-1] in punct_dict:
                        word_tokens.append(token[:-1])
                        word_tokens.append(token[-1])
                    else:
                        word_tokens.append(token)

            self.word_token_list.append(word_tokens)
            self.word_freq_dict += Counter(word_tokens)

        # Generate word and character indices
        self.num_words = len(self.word_freq_dict)

        if self.vocabulary_size == 0:
            self.vocabulary_size = None
        # check when most common
        self.word_list = [key for key, count in self.word_freq_dict.most_common(self.vocabulary_size - 1)]  # set the parameter to 512 4096
        self.vocab_id_dict = {word: index for index, word in enumerate(self.word_list)}
        self.id_vocab_dict = {index: word for index, word in enumerate(self.word_list)}
        self.vocab_id_dict['UNKNOWN'] = self.vocabulary_size - 1
        self.vocab_id_dict[self.vocabulary_size - 1] = 'UNKNOWN'
        # fix the loop
        for document in self.document_list:
            numeric_document = []
            for token in document:
                if token in self.vocab_id_dict:
                    numeric_document.append(self.vocab_id_dict[token])
                else:
                    numeric_document.append(self.vocabulary_size - 1)
            self.numeric_document_list.append(numeric_document)

        if self.create_char_corpus:
            self.num_characters = len(self.character_freq_dict)
            self.character_list = [key for key, count in self.character_freq_dict.most_common()]
            self.character_index_dict = {word: index for index, word in enumerate(self.character_list)}

    def get_numeric_document_lists(self):
        """
        Loads document titles and their content from the specified directory.
        """
        document_list = []
        document_title_list = []
        directory_list = os.listdir(self.input_path)
        for file_name in directory_list:
            if file_name[0] != ".":
                document_title_list.append(os.path.splitext(file_name)[0])
                with open(self.input_path+file_name, "r") as f:
                    file_string = f.read()
                document_list.append(file_string)
        combined_list = list(zip(document_title_list, document_list))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        self.document_list = [x[1] for x in sorted_combined_list]
        self.document_title_list = [x[0] for x in sorted_combined_list]

    def set_num_documents(self):
        """
        Sets the number of documents to be considered for the corpus.
        """
        if self.num_documents is None:
            self.num_documents = len(self.document_list)
        else:
            if self.num_documents > len(self.document_list):
                self.num_documents = len(self.document_list)
            else:
                self.document_list = self.document_list[:self.num_documents]
                self.document_title_list = self.document_title_list[:self.num_documents]

    @staticmethod
    def preprocess_document(text):
        """
        Preprocesses the document text by converting to lowercase and handling newlines.
        """
        text = text.lower()
        sentence_end_chars = r"\,(\.|!|\?)"
        text = re.sub(r"(?<!" + sentence_end_chars + r")(?<!\n)\n(?!\n)", " ", text)
        return text

    @staticmethod
    def get_sorted_unique_token_list(freq_dict):
        """
        Returns a sorted list of unique tokens based on their frequency.
        """
        return sorted(freq_dict, key=lambda x: freq_dict[x], reverse=True)

    def save_corpus(self, output_path):
        """
        Saves the corpus to a specified path.
        """
        print("Saving corpus")
        with open(output_path+".pkl", "wb") as f:
            pickle.dump(self, f)

    def load_corpus(self, input_path):
        """
        Loads a corpus from a specified path.
        """
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        for key, value in vars(data).items():
            setattr(self, key, value)