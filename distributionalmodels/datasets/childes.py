from distributionalmodels.datasets.corpus import Corpus
import pandas as pd
import os


class Childes(Corpus):

    def __init__(self):
        super().__init__()

        self.input_path = None
        self.language = None
        self.age_range_tuple = None
        self.sex_list = None
        self.collection_name = None
        self.add_punctuation = None
        self.exclude_target_child = None

    def download_from_childes_db(self):
        # TODO Implement this
        pass

    def load_corpus_info(self):

        corpus_info_file_path = os.path.join(self.input_path, "raw_childes.csv")
        corpus_info_df = pd.read_csv(corpus_info_file_path, header=0)
        corpus_info_df = corpus_info_df[["language", "collection_name", "transcript_id", "target_child_id", "target_child_age", "target_child_sex"]]
        corpus_info_df = corpus_info_df.sort_values(by=['transcript_id'])
        corpus_info_df = corpus_info_df.reset_index()
        return corpus_info_df

    def load_utterance_df(self):

        header_list = ["id",
                       "gloss",
                       "stem",
                        "actual_phonology",
                        "model_phonology",
                        "type",
                        "language",
                        "num_morphemes",
                        "num_tokens",
                        "utterance_order",
                        "corpus_name",
                        "part_of_speech",
                        "speaker_code",
                        "speaker_name",
                        "speaker_role",
                        "target_child_name",
                        "target_child_age",
                        "target_child_sex",
                        "media_start",
                        "media_end",
                        "media_unit",
                        "collection_name",
                        "collection_id",
                        "corpus_id",
                        "speaker_id",
                        "target_child_id",
                        "transcript_id"]

        utterance_file_path = os.path.join(self.input_path, "raw_childes.csv")
        utterance_df = pd.read_csv(utterance_file_path, header=0)

        utterance_df = utterance_df.dropna(subset=['gloss'])

        if self.language is not None:
            utterance_df = utterance_df[utterance_df['language'] == self.language]
        if self.collection_name is not None:
            utterance_df = utterance_df[utterance_df['collection'] == self.collection_name]
        if self.sex_list is not None:
            utterance_df = utterance_df[utterance_df['target_child_sex'].isin(self.sex_list)]
        if self.age_range_tuple is not None:
            utterance_df = utterance_df.dropna(subset=['target_child_age'])
            utterance_df = utterance_df[(utterance_df['target_child_age'] >= self.age_range_tuple[0]) & (utterance_df['target_child_age'] <= self.age_range_tuple[1])]
        if self.exclude_target_child:
            utterance_df = utterance_df[utterance_df["speaker_role"] != "Target_Child"]

        utterance_df = utterance_df.sort_values(by=["target_child_age", "transcript_id", "utterance_order"], ascending=[True, True, True])

        return utterance_df

    def create_corpus(self,
                      input_path,
                      language=None,
                      age_range_tuple=None,
                      sex_list=None,
                      collection_name=None,
                      add_punctuation=True,
                      exclude_target_child=True,
                      get_spacy_tokens=True,
                      num_docs=None,
                      create_char_corpus=False):

        self.input_path = input_path
        self.num_documents = num_docs
        self.language = language
        self.age_range_tuple = age_range_tuple
        self.sex_list = sex_list
        self.collection_name = collection_name
        self.add_punctuation = add_punctuation
        self.exclude_target_child = exclude_target_child
        self.get_spacy_tokens = get_spacy_tokens
        self.create_char_corpus = create_char_corpus

        self.get_numeric_document_lists()
        self.set_num_documents()
        self.process_numeric_document_list()

    def get_numeric_document_lists(self):
        utterance_df = self.load_utterance_df()

        punctuation_dict = {'declarative': ".",
                            "question": "?",
                            "trail off": ";",
                            "imperative": "!",
                            "imperative_emphatic": "!",
                            "interruption": ":",
                            "self interruption": ";",
                            "quotation next line": ";",
                            "interruption question": "?",
                            "missing CA terminator": ".",
                            "broken for coding": ".",
                            "trail off question": "?",
                            "quotation precedes": ".",
                            "self interruption question": "?",
                            "question exclamation": "?"}

        document_tuple_list = []

        for transcript_id, group in utterance_df.groupby('transcript_id'):
            transcript_df = group.copy()
            gloss_list = transcript_df['gloss'].tolist()
            type_list = transcript_df['type'].tolist()
            age_list = transcript_df['target_child_age'].tolist()
            utterance_order_list = transcript_df['utterance_order'].tolist()
            num_utterances = len(utterance_order_list)
            utterance_tuple_list = [(utterance_order_list[i], gloss_list[i], type_list[i]) for i in range(num_utterances)]
            utterance_tuple_list.sort()

            transcript_string_list = []
            for i in range(num_utterances):
                utterance_tuple = utterance_tuple_list[i]
                new_string = utterance_tuple[1]
                utterance_type = utterance_tuple[2]
                if self.add_punctuation:
                    new_string += punctuation_dict[utterance_type]
                transcript_string_list.append(new_string)
            document_string = " ".join(transcript_string_list)
            document_tuple_list.append((age_list[0], document_string, transcript_id))

        self.document_list = []
        self.document_title_list = []
        document_tuple_list.sort()
        for document_tuple in document_tuple_list:
            self.document_list.append(document_tuple[1])
            age = document_tuple[0]
            transcript_id = document_tuple[2]
            document_title = str(round(age, 3)).replace('.', '_') + "_" + str(transcript_id)
            self.document_title_list.append(document_title)

