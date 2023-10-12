from distributionalmodels.datasets.dataset import Dataset
class Childes(Dataset):

    def __init__(self):
        super().__init__()

        # what are the exact data structures the model expects

        # load the file, and create the data structures from that file

        # we need a function

        self.unknown_token = "__UNK__"

    def create_vocab(self, n=0, include_file=None, exclude_file=None, use_unknown=True):

        # n = 4096
        if use_unknown:
            n -= 1

        # start with the list words in include_file that are in the corpus
        # count = len(
        # while
        exclude_list = []
        with open(exclude_file) as f:
            for line in f:
                vocab_item = line.strip('\n').strip()
                exclude_list.append(vocab_item)

        self.vocab_list = []
        with open(include_file) as f:
            for line in f:
                vocab_item = line.strip('\n').strip()
                if vocab_item in self.corpus_master_vocab_dict:
                    self.vocab_list.append(vocab_item)

        count = len(self.vocab_list)
        while count < n:
            for vocab_item in self.freq_sorted_master_vocab_list:
                if vocab_item not in self.vocab_list:
                    if vocab_item not in exclude_list:
                        self.vocab_list.append(vocab_item)
                        count += 1

        if use_unknown:
            self.vocab_list.append(self.unknown_token)

    def replace_unknowns(self):
        for i, word in enumerate(self.corpus_list):
            if word not in self.vocab_list:
                self.corpus_list[i] == self.unknown_token


