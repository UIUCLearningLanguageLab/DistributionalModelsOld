from semanticmodels2.tasks.task import Task
import numpy as np
import pandas as pd
import time as time
from statistics import mean
import csv
from scripts.utils import *

class CohyponymTask(Task):
    def __init__(self, category_info, score_matrix, data_index_dict, categories_from_file=False, num_thresholds=9):

        '''
            threshold = [1, .8, .6, .4, .2, 0]
            data matrix is a matrix of values comparing all words to each other, each score must be between 0 and 1
            lion-tiger = .95
            lion-deer = .6

            go through each pair of words
            compare their score to some threshold
                if score > threshold:
                    decide "same category"
                else:
                    decide "different category"
        '''

        super().__init__()
        self.category_info = category_info  # csv file with category,target pairs. No duplicate targets allowed, no header
        self.data_index_dict = data_index_dict  # labels and indexes for the data matrix
        self.score_matrix = score_matrix  # a square matrix of values between 0 and 1 relating all potential targets

        self.num_thresholds = num_thresholds  # number of thresholds to test

        self.num_categories = 0  # num of categories in the category file
        self.category_list = []  # list of categories
        self.category_index_dict = {}  # unique index for each category
        self.categories_from_file = categories_from_file

        self.num_targets = 0  # num unique targets in category file
        self.target_list = []  # list of unique targets
        self.target_index_dict = {}  # unique index for each target
        self.target_matrix_index_dict = {} # index for the target in the matrix

        self.target_category_dict = {}  # dictionary of targets pointing to their category deer: herbivore
        self.category_target_list_dict = {}  # dictionary of categories pointing to list of targets in that category herbivore: [deer, elephant]

        self.guess_accuracy_df = None
        self.guess_accuracy_df_list = []
        self.target_balanced_accuracy_df = None
        self.target_accuracy_df = None

        self.mean_category_score_matrix = np.zeros([self.num_categories, self.num_categories])
        self.confusion_matrix = np.zeros([self.num_categories, self.num_categories])

        print("Computing Balanced Accuracy")
        self.threshold_list = np.linspace(-1, 1, self.num_thresholds)
        self.load_categories()
        self.create_results_df()
        self.compute_ba()

    def load_categories(self):
        if self.categories_from_file:
            category_info = {}
            with open(self.category_info, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    category_info[row[0]] = row[1]
            self.category_info = category_info

        for target, category in self.category_info.items():
            if target in self.target_list:
                raise Exception("ERROR: Target {} is duplicated in category file".format(target))
            else:
                if target in self.data_index_dict:
                    self.target_list.append(target)
                    self.target_index_dict[target] = self.num_targets
                    self.num_targets += 1
                    self.target_category_dict[target] = category
                    self.target_matrix_index_dict[target] = self.data_index_dict[target]

                    if category not in self.category_index_dict:
                        self.category_list.append(category)
                        self.category_index_dict[category] = self.num_categories
                        self.num_categories += 1
                        self.category_target_list_dict[category] = []
                    self.category_target_list_dict[category].append(target)
                else:
                    print(f"    WARNING: Target {target} not in similarity matrix so ignoring it")

    def create_results_df(self):

        results = []
        for i in range(self.num_thresholds):
            threshold = self.threshold_list[i]

            for j in range(self.num_targets):
                target1 = self.target_list[j]
                category1 = self.target_category_dict[target1]

                for k in range(self.num_targets):
                    target2 = self.target_list[k]
                    category2 = self.target_category_dict[target2]
                    score = self.score_matrix[j, k]

                    if -1 > score < 1:
                        raise ValueError(f"ERROR: Matrix score {j}{k}:{score} out of bounds 0-1")

                    if score > threshold:
                        guess = 1
                    else:
                        guess = 0

                    if category1 == category2:
                        actual = 1
                    else:
                        actual = 0

                    if guess == actual:
                        correct = 1
                        if guess == 1:
                            sd_result = "hit"
                        else:
                            sd_result = "cr"
                    else:
                        correct = 0
                        if guess == 0:
                            sd_result = "miss"
                        else:
                            sd_result = "fa"

                    data = [threshold,
                            target1,
                            target2,
                            category1,
                            category2,
                            j,
                            k,
                            score,
                            guess,
                            actual,
                            correct,
                            sd_result]
                    results.append(data)

        self.guess_accuracy_df = pd.DataFrame(results, columns=['threshold',
                                                                'target1',
                                                                'target2',
                                                                'category1',
                                                                'category2',
                                                                'target1_index',
                                                                'target2_index',
                                                                'score',
                                                                'guess',
                                                                'actual',
                                                                'correct',
                                                                'sd_result'])

    def create_average_similarity_score_matrix(self):
        df = self.guess_accuracy_df.loc[
            (self.guess_accuracy_df['threshold'] == self.threshold_list[0]), ["target1", "target2", "category1",
                                                                            "category2", "score"]]
        df = df.reset_index(drop=True)
        for category1, index1 in self.category_index_dict.items():
            for category2, index2 in self.category_index_dict.items():
                indices = df.index[
                    (df['category1'] == category1) & (df['category2'] == category2) & (df["target1"] != df["target2"])]
                if indices.tolist():
                    temp_similarity_scores = df.loc[indices]['score'].tolist()
                    similarity_scores = list(set(temp_similarity_scores))
                    ave_sim_score = mean(similarity_scores)
                    self.mean_category_score_matrix[index1, index2] = ave_sim_score

    def compute_confusion_matrix(self):
        pass

    def compute_ba(self, exclude_identity_pairs=False, exclude_symmetric_pairs=False):


        if exclude_identity_pairs:
            self.guess_accuracy_df = self.guess_accuracy_df[
                self.guess_accuracy_df['target1'] != self.guess_accuracy_df['target2']]
        if exclude_symmetric_pairs:
            self.guess_accuracy_df = self.guess_accuracy_df[
                self.guess_accuracy_df['target1_index'] < self.guess_accuracy_df['target2_index']]
        self.guess_accuracy_df = self.guess_accuracy_df.applymap(lambda x: round_and_check(x, 2))

        # Group by threshold and target1_index, and calculate the mean and standard deviation of correct for each group
        self.target_accuracy_df = self.guess_accuracy_df.groupby(['threshold',
                                                                'target1',
                                                                'category1',
                                                                'actual'])['correct'].agg(['mean',
                                                                                            'std',
                                                                                            'count']).reset_index()

        self.target_accuracy_df.columns = ['threshold', 'target', 'category', 'actual', 'mean_correct',
                                        'stdev_correct', 'n']

        self.target_ba_df = self.target_accuracy_df.groupby(['threshold', 'target', 'category'])['mean_correct'].mean().reset_index()
        self.target_ba_df.columns = ['threshold', 'target', 'category', 'ba']

        self.category_ba_df = self.target_ba_df.groupby(['threshold', 'category'])['ba'].agg(['mean', 'std', 'count']).reset_index()
        self.category_ba_df.columns = ['threshold', 'category', 'mean_ba', "std_ba", 'n']
        self.category_ba_df['stderr_ba'] = self.category_ba_df['std_ba'] / np.sqrt(self.category_ba_df['n'])

        self.threshold_ba_df = self.category_ba_df.groupby(['threshold'])['mean_ba'].mean().reset_index()
        self.threshold_ba_df.columns = ['threshold', 'mean_ba']

        # # Get the index of the row with the highest value of mean_ba in threshold_ba_df
        max_mean_ba_index = self.threshold_ba_df['mean_ba'].idxmax()
        max_mean_ba_threshold = self.threshold_ba_df.loc[max_mean_ba_index, 'threshold']
        #
        self.best_target_ba_df = self.target_ba_df[self.target_ba_df['threshold'] == max_mean_ba_threshold].reset_index()
        self.best_category_ba_df = self.best_target_ba_df.groupby(['category', 'threshold'])['ba'].agg(['mean', 'std', 'count']).reset_index()
        self.best_category_ba_df.columns = ['category', 'best_overall_threshold', 'ba_mean', 'ba_std', 'n']
        self.best_category_ba_df = self.best_category_ba_df.applymap(lambda x: round_and_check(x, 2))
        self.guess_accuracy_best_df = self.guess_accuracy_df[self.guess_accuracy_df['threshold']==self.best_category_ba_df['best_overall_threshold'].unique()[0]]