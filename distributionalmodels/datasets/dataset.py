from typing import Dict, List
import os
class Dataset:

    def __init__(self):

        self.vocab_index_dict = None
        self.index_vocab_dict = None
        self.numeric_document_list = None
        self.numeric_document_sentence_list = None


    def export_corpus_as_file(self, path):
        decode = lambda l: [self.id_vocab_dict[i] for i in l]
        # with open("datasets/{}_corpus.txt".format(corpus_name), 'w') as f:
        for idx, doc in enumerate(self.numeric_document_sentence_list):
            # Create a unique filename for each document.
            filename = os.path.join(str(path), f'document_{idx + 1}.csv')
            # Open a new CSV file for writing.
            with open(filename, mode='w', newline='') as f:
                for document in self.numeric_document_sentence_list:
                    for sentence in document:
                        f.write(str(decode(sentence)))
                        f.write("\n")
            


