import sys
sys.path.append("/Users/jingfengzhang/FirstYearProject/DistributionalModels")
from distributionalmodels.datasets import childes
import os


def main():
    language = "eng"
    collection_name = None
    age_range_tuple = (0, 1000)
    sex_list = None
    add_punctuation = False
    exclude_target_child = True
    get_spacy_tokens = False
    num_docs = None
    create_char_corpus = False

    new_corpus = childes.Childes()
    # path = os.getcwd()
    # print(path)
    # if os.path.exists(path):
    #     print(f"The path {path} exists.")
    # else:
    #     print(f"The path {path} does not exist.")
    new_corpus.create_corpus("/Users/jingfengzhang/FirstYearProject/DistributionalModels/corpus_info/childes",
                             language=language,
                             collection_name=collection_name,
                             age_range_tuple=age_range_tuple,
                             sex_list=sex_list,
                             add_punctuation=add_punctuation,
                             exclude_target_child=exclude_target_child,
                             get_spacy_tokens=get_spacy_tokens,
                             num_docs=num_docs,
                             create_char_corpus=create_char_corpus)

    new_corpus.save_corpus("corpora/childes")


if __name__ == '__main__':
    main()

