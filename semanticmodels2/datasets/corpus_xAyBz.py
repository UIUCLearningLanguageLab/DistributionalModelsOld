import string
import random
import copy
from venv import create
from semanticmodels2.datasets.dataset import Dataset
from typing import List, Tuple, Dict
import time
import os
import itertools

random.seed(1023)


class Corpus_xAyBz(Dataset):

    def __init__(self,
                 num_AB_categories=2,
                 AB_category_size=3,

                 x_category_size=2,
                 y_category_size=2,
                 z_category_size=2,
                 min_x_per_sentence=1,
                 max_x_per_sentence=1,
                 min_y_per_sentence=1,
                 max_y_per_sentence=1,
                 min_z_per_sentence=1,
                 max_z_per_sentence=1,

                 document_organization_rule='single_sentence',
                 document_repetitions=4,
                 document_sequence_rule='massed',

                 sentence_repetitions_per_document=4,
                 sentence_sequence_rule='massed',

                 word_order_rule='fixed',
                 include_punctuation=True,

                 random_seed=None
                 ):
        super().__init__()

        '''
        Create documents composed sentences, which are composed of words.
        Words belong to syntactic categories A,B,x,y,z denoting their sentence position
        Sentences are constructed using the following rules:
            S = (XP) + Ai + (YP) + Bi + (ZP) .
            XP = (XP) + x
            YP = (YP) + y
            ZP = (ZP) + z
            and where i refers to specific and linked A and B sub-categories
        Results in sentences of the form:
            [x1, x2, ..., xn, A, y1, y2, ..., yn, B, z1, z2, ..., z3 .]

        All A and B sub-categories are sets of identical size
        A and B words are named with the following conventions:
            First character is always A or B, denoting sentence position
            Second character is a number denoting the sub-category
            Third character is a number denoting which the word's index within the sub-category
            Example: A32
                - occupies the first syntactic spot in the pair (because of the A),
                - belongs to the third sub-category,
                - and is the second word in the third category

        A legal sentence, can pair any words from matching A and B sub-categories EXCEPT for those that
        have the same index within that category.
        In other words:
            A11-B12 and A33-B35 are legal pairs,
            A11-B11 is NOT a legal pair, because the words share the same index within their subcategories
            A11-B22 is NOT a legal pair, because the words belong to different sub-categories



            Example:
                num_AB_categories = 3
                AB_category_size = 4
                x_category_size = 4
                y_category_size = 8
                z_category_size = 4

                AB_category_list = [1,2,3]
                AB_category_dict = {1: [[A11, A12, A13, A14], [B11, B12, B13, B14]],
                                    2: [[A21, A22, A23, A24], [B21, B22, B23, B24]],
                                    3: [[A31, A32, A33, A34], [B31, B32, B33, B34]]}

                included_AB_pair_list = [[A11, B12], [A11, B13], [A11, B14],
                                         [A12, B11], [A12, B13], [A12, B14],
                                         [A13, B11], [A13, B12], [A13, B14],
                                         [A14, B11], [A14, B12], [A14, B13],
                                         [A21, B22], [A21, B23], [A21, B24],
                                         [A22, B21], [A22, B23], [A22, B24],
                                         [A23, B21], [A23, B22], [A23, B24],
                                         [A24, B21], [A24, B22], [A24, B23],
                                         [A31, B32], [A31, B33], [A31, B34],
                                         [A32, B31], [A32, B33], [A32, B34],
                                         [A33, B31], [A33, B32], [A33, B34],
                                         [A34, B31], [A34, B32], [A34, B33]]

                ommitted_AB_pair_list = [[A11, B11], [A12, B12], [A13, B13], [A14, B14],
                                         [A21, B21], [A22, B22], [A23, B23], [A24, B24],
                                         [A31, B31], [A32, B32], [A33, B33], [A34, B34]]

                x_list = [x1, x2, x3, x4]
                y_list = [y1, y2, y3, y4, y5, y6, y7, y8]
                z_list = [z1, z2, z3, z4]

        There are a deterministic number of sentences with unique target pairs, determined jointly by
            category size and num categories and equal to:
            num_unique_target_sentences = target_category_size*(target_category_size-1)*num_target_categories

        These sentences are distributed into documents according to document_organization_rule, with allowed values:
            - "all_pairs": each document contains all possible sentence pairs, resulting in a single base document
            - "one_pair_each_category": each document contains a single sentence from each category, resulting in the number
                of base documents equaling: num_sentences / num_target_categories
            - "single_sentence": each document contains a single sentence, resulting in num_unique_target_sentences base documents
            - "single_category": each document contains all sentences from a single category,
                resulting in num_target_categories base documents

        Within each document, its sentences are repeated sentence_repetitions_per_document number of times, each time with different
            random interveners, according to the rules described below.
        The sentences are organized by one of three rules: massed (which puts all sentences next to each other which share the same A word),
            interleaved (which rotates the sentences by A word), and shuffled (which randomizes the order with regard to A word)

        Each base document is repeated document_repetitions number of times, each time with different random interveners, according
            to the rules described below.

        The documents are ordered according to the document_sequence_rule, which must be either 'massed' or 'interleaved'
            - massed: the documents that share the common property (word, category, etc.) are adjacent in the document sequence order
                e.g. [A1,A1,A2,A2,A3,A3]
            - interleaved: the documents that share the common property are distributed in the document sequence order
                e.g. [A1,A2,A3,A1,A2,A3]

        Word order within sentences follows one of three rules:
            - random: completely randomized
            - targets_fixed: targets

        Sentences can either end with puncutation "." or have that omitted
        '''

        self.document_list = None
        self.document_group_list = None
        self.num_documents = None

        self.document_organization_rule = document_organization_rule
        self.document_repetitions = document_repetitions
        self.document_sequence_rule = document_sequence_rule

        self.num_AB_categories = num_AB_categories
        self.AB_category_size = AB_category_size
        self.AB_category_dict = None

        self.x_category_size = x_category_size
        self.min_x_per_sentence = min_x_per_sentence
        self.max_x_per_sentence = max_x_per_sentence
        self.x_list = None

        self.y_category_size = y_category_size
        self.min_y_per_sentence = min_y_per_sentence
        self.max_y_per_sentence = max_y_per_sentence
        self.y_list = None

        self.z_category_size = z_category_size
        self.min_z_per_sentence = min_z_per_sentence
        self.max_z_per_sentence = max_z_per_sentence
        self.z_list = None

        self.sentence_repetitions_per_document = sentence_repetitions_per_document
        self.sentence_sequence_rule = sentence_sequence_rule

        self.word_order_rule = word_order_rule
        self.include_punctuation = include_punctuation

        self.random_seed = random_seed
        random.seed(random_seed)

        self.vocabulary_size = None
        self.included_AB_pair_list = None
        self.omitted_AB_pair_list = None

        self.check_parameters()
        self.create_corpus_name()
        start_time = time.time()
        self.create_vocabulary()
        self.create_word_pair_list()
        self.create_documents()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time for creating corpus_xAyBz is :', elapsed_time, 'seconds')
        self.create_numeric_token_sequence_list()

    # self.create_syntagmatic_category_file(category_list=["A","B"])
    # self.export_corpus_as_file(self.corpus_name)

    def create_corpus_name(self):
        self.corpus_name = "Corpus"
        self.corpus_name += "_" + str(self.num_AB_categories)
        self.corpus_name += "_" + str(self.AB_category_size)
        self.corpus_name += "_" + str(self.x_category_size)
        self.corpus_name += "_" + str(self.min_x_per_sentence)
        self.corpus_name += "_" + str(self.max_x_per_sentence)
        self.corpus_name += "_" + str(self.y_category_size)
        self.corpus_name += "_" + str(self.min_y_per_sentence)
        self.corpus_name += "_" + str(self.max_y_per_sentence)
        self.corpus_name += "_" + str(self.z_category_size)
        self.corpus_name += "_" + str(self.min_z_per_sentence)
        self.corpus_name += "_" + str(self.max_z_per_sentence)
        self.corpus_name += "_" + str(self.document_organization_rule)
        self.corpus_name += "_" + str(self.document_repetitions)
        self.corpus_name += "_" + str(self.document_sequence_rule)
        self.corpus_name += "_" + str(self.sentence_repetitions_per_document)
        self.corpus_name += "_" + str(self.sentence_sequence_rule)

        self.corpus_name += "_" + str(self.random_seed)

        ("\nCreating corpus {}".format(self.corpus_name))

    def __repr__(self):
        output_string = "\n{}\n".format(self.corpus_name)
        output_string += "    Vocab Size: {}\n".format(self.vocabulary_size)
        output_string += "    Punctuation: [.]\n"
        output_string += "	  x Category: [{}]\n".format(",".join(self.x_list))
        output_string += "	  y Category: [{}]\n".format(",".join(self.y_list))
        output_string += "	  z Category: [{}]\n".format(",".join(self.z_list))

        for category, members in self.AB_category_dict.items():
            c1 = members[0]
            c2 = members[1]
            output_string += "	  {}: [{}]\n".format("A" + category, ",".join(c1))
            output_string += "	  {}: [{}]\n".format("B" + category, ",".join(c2))

        output_string += "    Documents:\n"
        for i in range(self.num_documents):
            # output_string += "        {} Group:{} Size:{} | [{}]\n".format(i, self.document_group_list[i], len(self.document_list), ",".join(self.document_list[i]))
            output_string += "        Document:{} Group:{} Len:{}\n".format(i, self.document_group_list[i],
                                                                            len(self.document_list[i]))
            for j in range(len(self.document_list[i])):
                output_string += "            [{}]\n".format(",".join(self.document_list[i][j]))

        output_string += "\n"
        return output_string

    def check_parameters(self):
        if self.num_AB_categories < 1:
            raise Exception("ERROR: num_AB_categories must be between >=1".format())
        if self.AB_category_size < 2:
            raise Exception("ERROR: AB_category_size must be >= 2")
        if self.x_category_size < 0:
            raise Exception("ERROR: x_category_size must be >= 1")
        if self.min_x_per_sentence < 0:
            raise Exception("ERROR: min_x_per_sentence must be >= 0")
        if self.min_x_per_sentence > self.max_x_per_sentence:
            raise Exception("ERROR: min_x_per_sentence must be <= max_x_per_sentence")
        if self.max_x_per_sentence < 0:
            raise Exception("ERROR: max_x_per_sentence must be >= 0")
        if self.y_category_size < 0:
            raise Exception("ERROR: y_category_size must be >= 1")
        if self.min_y_per_sentence < 0:
            raise Exception("ERROR: min_y_per_sentence must be >= 0")
        if self.min_y_per_sentence > self.max_y_per_sentence:
            raise Exception("ERROR: min_y_per_sentence must be <= max_y_per_sentence")
        if self.max_y_per_sentence < 0:
            raise Exception("ERROR: max_y_per_sentence must be >= 0")
        if self.z_category_size < 0:
            raise Exception("ERROR: z_category_size must be >= 1")
        if self.min_z_per_sentence < 0:
            raise Exception("ERROR: min_z_per_sentence must be >= 0")
        if self.min_z_per_sentence > self.max_z_per_sentence:
            raise Exception("ERROR: min_z_per_sentence must be <= max_z_per_sentence")
        if self.max_z_per_sentence < 0:
            raise Exception("ERROR: max_z_per_sentence must be >= 0")
        # if self.sentence_repetitions_per_document < 1:
        # 	raise Exception("ERROR: sentence_repetitions_per_document must be >= 1")
        if self.document_repetitions < 1:
            raise Exception("ERROR: document_repetitions must be >= 1")
        if self.document_organization_rule not in ["all_pairs", "one_pair_each_category", "single_sentence",
                                                   "single_category"]:
            raise Exception("ERROR: Unrecognized document organization rule {}".format(self.document_organization_rule))
        if self.document_sequence_rule not in ["massed", "interleaved", 'random']:
            raise Exception("ERROR: Unrecognized document_sequence_rule {}".format(self.document_sequence_rule))

    def create_category_members(self, size, label):
        member_list = []
        if size > 0:
            for i in range(size):
                member = label + str(i + 1)
                member_list.append(member)
        return member_list

    def add_words_to_vocab(self, word_list):
        if len(word_list) > 0:
            for word in word_list:
                self.vocab_id_dict[word] = self.vocabulary_size
                self.id_vocab_dict[self.vocabulary_size] = word
                self.type_list.append(word)
                self.vocabulary_size += 1

    def create_vocabulary(self):
        self.type_list = []
        self.AB_category_dict = {}
        self.vocab_id_dict = {}
        self.id_vocab_dict = {}
        self.vocabulary_size = 0

        # self.add_words_to_vocab(['PAD'])
        self.add_words_to_vocab(['.'])

        for i in range(self.num_AB_categories):
            current_category = str(i + 1)

            category_label = "A" + current_category + "_"
            set1 = self.create_category_members(self.AB_category_size, category_label)
            self.add_words_to_vocab(set1)

            category_label = "B" + current_category + "_"
            set2 = self.create_category_members(self.AB_category_size, category_label)
            self.add_words_to_vocab(set2)

            self.AB_category_dict[current_category] = [set1, set2]

        self.x_list = self.create_category_members(self.x_category_size, "x")
        self.add_words_to_vocab(self.x_list)

        self.y_list = self.create_category_members(self.y_category_size, "y")
        self.add_words_to_vocab(self.y_list)

        self.z_list = self.create_category_members(self.z_category_size, "z")
        self.add_words_to_vocab(self.z_list)

    def create_word_pair_list(self):
        self.included_AB_pair_list = []
        self.omitted_AB_pair_list = []

        for category, data in self.AB_category_dict.items():
            set1 = data[0]
            set2 = data[1]
            for i in range(self.AB_category_size):
                for j in range(self.AB_category_size):
                    if i == j:
                        self.omitted_AB_pair_list.append((set1[i], set2[j]))
                    else:
                        self.included_AB_pair_list.append((set1[i], set2[j]))

    def get_pair_document_group(self, pair, index):
        if self.document_organization_rule == 'all_pairs':
            group = "all"
        elif self.document_organization_rule == 'one_pair_each_category':
            group = str(round(index % (len(self.included_AB_pair_list) / self.num_AB_categories)))
        elif self.document_organization_rule == 'single_sentence':
            group = "_".join(pair)
        elif self.document_organization_rule == 'single_category':
            group = pair[0][1]
        else:
            raise Exception("ERROR: unrecognized document_organization_rule {}".format(self.document_organization_rule))
        return group

    def create_documents(self):
        document_group_dict = {}
        included_AB_pair_list = copy.deepcopy(self.included_AB_pair_list)
        group_list = []
        for i in range(len(included_AB_pair_list)):
            pair = included_AB_pair_list[i]
            group = self.get_pair_document_group(pair, i)
            if not group in document_group_dict:
                document_group_dict[group] = []
                group_list.append(group)
            document_group_dict[group].append(pair)

        group_size = len(document_group_dict[group])
        num_groups = len(document_group_dict)

        full_document_list = []
        full_document_group_list = []

        if self.document_sequence_rule == "massed":
            for i in range(num_groups):
                group = group_list[i]
                for j in range(self.document_repetitions):
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)

        elif self.document_sequence_rule == "interleaved":
            for i in range(self.document_repetitions):
                for j in range(num_groups):
                    group = group_list[j]
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)

        elif self.document_sequence_rule == "random":
            # TODO impliment this
            for i in range(num_groups):
                group = group_list[i]
                for j in range(self.document_repetitions):
                    document = copy.deepcopy(document_group_dict[group])
                    full_document_list.append(document)
                    full_document_group_list.append(group)
            random.shuffle(full_document_list)

        else:
            raise Exception("ERROR: Unrecognized document_sequence_rule {}".format(self.document_sequence_rule))

        self.document_list = []
        self.document_group_list = []

        # if sentence reps per doc == 0, then we are going to do the number of reps requred to do all combos of ys
        # else do a fixed number of reps and choose a random y
        if self.sentence_repetitions_per_document == 0:
            y_lists = []
            for i in range(self.min_y_per_sentence, self.max_y_per_sentence + 1):
                y_lists += list(itertools.product(self.y_list, repeat=i))

        for i in range(len(full_document_list)):
            new_document = []
            current_document_template = copy.deepcopy(full_document_list[i])
            num_sentences = len(current_document_template)

            if self.sentence_repetitions_per_document > 0:

                if self.sentence_sequence_rule == "massed" or self.sentence_sequence_rule == "random":
                    for j in range(num_sentences):
                        current_AB_pair = current_document_template[j]
                        for k in range(self.sentence_repetitions_per_document):
                            sentence = self.create_sentence(current_AB_pair)
                            new_document.append(sentence)

                    if self.sentence_sequence_rule == "random":
                        random.shuffle(new_document)

                elif self.sentence_sequence_rule == "interleaved":
                    for j in range(self.sentence_repetitions_per_document):
                        for k in range(num_sentences):
                            current_AB_pair = current_document_template[k]
                            sentence = self.create_sentence(current_AB_pair)
                            new_document.append(sentence)
                else:
                    raise Exception("ERROR: unrecognized sentence_sequence_rule={}".format(self.sentence_sequence_rule))

                self.document_list.append(new_document)
                self.document_group_list.append(full_document_group_list[i])

            if self.sentence_repetitions_per_document == 0:

                if self.sentence_sequence_rule == "massed" or self.sentence_sequence_rule == "random":
                    for j in range(num_sentences):
                        current_AB_pair = current_document_template[j]
                        for k in range(len(y_lists)):
                            current_y_list = y_lists[k]
                            sentence = self.create_sentence(current_AB_pair, current_y_list)
                            new_document.append(sentence)

                    if self.sentence_sequence_rule == "random":
                        random.shuffle(new_document)

                elif self.sentence_sequence_rule == "interleaved":
                    for j in range(len(y_lists)):
                        current_y_list = y_lists[j]
                        for k in range(num_sentences):
                            current_AB_pair = current_document_template[k]
                            sentence = self.create_sentence(current_AB_pair, current_y_list)
                            new_document.append(sentence)

                self.document_list.append(new_document)
                self.document_group_list.append(full_document_group_list[i])

        self.num_documents = len(self.document_list)

    def create_sentence(self, AB_pair, current_y_list=None):
        sentence = []
        num_x = random.randint(self.min_x_per_sentence, self.max_x_per_sentence)
        num_y = random.randint(self.min_y_per_sentence, self.max_y_per_sentence)
        num_z = random.randint(self.min_z_per_sentence, self.max_z_per_sentence)
        for i in range(num_x):
            sentence.append(random.choice(self.x_list))

        sentence.append(AB_pair[0])

        if current_y_list == None:
            for i in range(num_y):
                sentence.append(random.choice(self.y_list))
        else:
            for y in current_y_list:
                sentence.append(y)

        sentence.append(AB_pair[1])

        for i in range(num_z):
            sentence.append(random.choice(self.z_list))

        if self.word_order_rule == "random":
            random.shuffle(sentence)

        if self.include_punctuation:
            sentence.append(".")
        return sentence

    def create_numeric_token_sequence_list(self):
        self.numeric_document_list = []
        self.numeric_document_sentence_list: List[List[List[int]]] = []
        self.sequence_list = []
        self.sequence_test_category_list: List[List[int]] = []
        for i in range(self.num_documents):
            sentence_list = []
            current_document = self.document_list[i]
            numeric_document = []
            sequence_document = []
            for j in range(len(current_document)):
                current_sentence = current_document[j]
                numeric_sentence = []
                A_word = ''
                for k in range(len(current_sentence)):
                    numeric_sentence.append(self.vocab_id_dict[current_sentence[k]])
                    if current_sentence[k][0] == 'A':
                        A_word = current_sentence[k]
                    if current_sentence[k][0] == 'B':
                        self.sequence_test_category_list.append(
                            self.assign_index_to_tokens(A_word, current_sentence[k]))
                    # print(current_sentence)
                    # print(self.vocab_dict.keys())
                    # print(self.assign_index_to_tokens(A_word, current_sentence[k]))
                sentence_list.append(numeric_sentence)
                numeric_document += numeric_sentence
                sequence_document += current_sentence
            self.numeric_document_list.append(numeric_document)
            self.numeric_document_sentence_list.append(sentence_list)
            self.sequence_list.append(sequence_document)

    def assign_index_to_tokens(self, A_word, B_word):
        sequence_category_list = []
        self.category_label_list = ['Period', 'Present A', 'Omitted A', 'Legal A', 'Illegal A', 'Present B',
                                    'Omitted B', 'Legal B', 'Illegal B', 'y']
        for word, index in self.vocab_id_dict.items():
            if word == '.':
                sequence_category_list.append(0)
            elif word[0] == 'A':
                if word == A_word:
                    sequence_category_list.append(1)
                elif word[1] == B_word[1]:
                    omitted_A = ''
                    for tuple in self.omitted_AB_pair_list:
                        if B_word in tuple:
                            omitted_A = tuple[0]
                    if word == omitted_A:
                        sequence_category_list.append(2)
                    else:
                        sequence_category_list.append(3)
                else:
                    sequence_category_list.append(4)
            elif word[0] == 'B':
                if word == B_word:
                    sequence_category_list.append(5)
                else:
                    if word[1] == A_word[1]:
                        omitted_B = ''
                        for tuple in self.omitted_AB_pair_list:
                            if A_word in tuple:
                                omitted_B = tuple[1]
                        if word == omitted_B:
                            sequence_category_list.append(6)
                        else:
                            sequence_category_list.append(7)
                    else:
                        sequence_category_list.append(8)
            else:
                sequence_category_list.append(9)

        return sequence_category_list

    def generate_paradigmatic_word_category_dict(self):
        temp_word_category_dict = {}
        for word in self.vocab_id_dict:
            if word != 'PAD':
                if word[0] == "A" or word[0] == 'B':
                    category = word[:2]
                else:
                    category = word[0]
                temp_word_category_dict[word] = category

        self.paradigmatic_word_category_dict = {word: category for word, category in
                                                sorted(temp_word_category_dict.items(), key=lambda item: item[0])}

    def create_paradigmatic_category_file(self, path):
        file_name = os.path.join(path, 'paradigmatic_category.csv')
        f = open(file_name, 'w')
        temp_word_category_dict = {}
        for word in self.vocab_id_dict:
            if word != 'PAD':
                if word[0] == "A" or word[0] == 'B':
                    category = word[:2]
                else:
                    category = word[0]
                temp_word_category_dict[word] = category

        self.paradigmatic_word_category_dict = {word: category for word, category in temp_word_category_dict.items()}
        for word, category in self.paradigmatic_word_category_dict.items():
            f.write("{},{}\n".format(word, category))
        f.close()

    def create_syntagmatic_category_file(self, path, category_list=["A", "B"], only_include_category=True):
        # TODO need to decide how we are going to impliment this
        '''
			Its a little complicated since its not as straightforward as saying "each word belongs to a category"
			For example, what are the correct relations for items in the A1 category? All the words B1, obviously.
			For paradigmatic, we check each word's category to see if they match: 
				so A1_1 and A1_2 are both A1, so correct answer is same
				and A1_1 and A2_1 have different A categories, so correct answer is "different"
			But for paradigmatic, we need to check if each A's number matches the B the number, 
				but we can't assign each A and B with matching numbers to the same category, because this procedure needs to count A1_1 and B1_2 as "same"
				but not count "A1_1 and A1_2"
			So we're going to have to write a different task function instead of our current co-hyponym task to deal with this, but we can worry about that later
			It will also have to figure out what to do with the x, y, and z words. They are also syntagmatic relations fo A1.
			There is a sense in which what we want to do is think of each word as a function that takes arguments, where the arguments are the sets of words that
				are each distinct legal kind of relationship
				So for A1_1(B1_list[:], x_list[:], y_list[:], z_list[:]), where :
		'''

        # TODO make a option whether include just A and B
        # f = open("datasets/syntagmatic_categories_{}.csv".format(self.corpus_name), 'w')
        file_name = os.path.join(path, 'syntagmatic_category.csv')
        f = open(file_name, 'w')
        if only_include_category:
            for word in self.vocab_id_dict:
                if word != "PAD":
                    category_list_copy = copy.deepcopy(category_list)
                    if word[0] in category_list:
                        target_category = word[:2]
                        for category in category_list:
                            if word[0] == category:
                                index = category_list.index(category)
                                category_list_copy.pop(index)
                                relation_category = ''
                                for i in range(len(category_list_copy)):
                                    relation_category += category_list_copy[i] + word[1]
                    else:
                        continue
                    f.write("{},{},{}\n".format(word, target_category, relation_category))
        else:
            for word in self.vocab_id_dict:
                if word != "PAD":
                    category_list_copy = copy.deepcopy(category_list)
                    if word[0] in category_list:
                        target_category = word[:2]
                        for category in category_list:
                            if word[0] == category:
                                index = category_list.index(category)
                                category_list_copy.pop(index)
                                relation_category = ''
                                for i in range(len(category_list_copy)):
                                    relation_category += category_list_copy[i] + word[1]
                    else:
                        target_category = word[0]
                        relation_category = target_category
                    f.write("{},{},{}\n".format(word, target_category, relation_category))
        f.close()


def main():
    # all these options need to be converted to params

    random_seed = None

    num_AB_categories = 3
    AB_category_size = 4

    x_category_size = 2
    min_x_per_sentence = 1
    max_x_per_sentence = 1

    y_category_size = 4
    min_y_per_sentence = 1
    max_y_per_sentence = 1

    z_category_size = 2
    min_z_per_sentence = 1
    max_z_per_sentence = 1

    document_repetitions = 1
    sentence_repetitions_per_document = 2

    document_organization_rule = "single_category"  # ["all_pairs", "one_pair_each_category", "single_sentence", "single_category"]
    document_sequence_rule = 'massed'  # ["massed", "interleaved", "random"]
    sentence_sequence_rule = 'massed'  # ["massed", "interleaved", "random"]

    word_order_rule = "fixed"
    include_punctuation = True

    corpus = Corpus_xAyBz(num_AB_categories, AB_category_size,
                          x_category_size, y_category_size, z_category_size,
                          min_x_per_sentence, min_y_per_sentence, min_z_per_sentence,
                          max_x_per_sentence, max_y_per_sentence, max_z_per_sentence,
                          document_organization_rule, document_repetitions, document_sequence_rule,
                          sentence_repetitions_per_document, sentence_sequence_rule,
                          word_order_rule, include_punctuation,
                          random_seed)
