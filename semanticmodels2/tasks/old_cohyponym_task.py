from semanticmodels2.tasks.task import Task
import numpy as np
import pandas as pd
import time as time
from statistics import mean

class Old_CohyponymTask(Task):

    def __init__(self, category_info, data_matrix, data_index_dict, num_thresholds=9):

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
        self.category_info = category_info          # csv file with category,target pairs. No duplicate targets allowed, no header
        self.data_index_dict = data_index_dict      # labels and indexes for the data matrix
        self.data_matrix = data_matrix              # a square matrix of values between 0 and 1 relating all potential targets
        self.num_thresholds = num_thresholds        # number of thresholds to test
        
        self.num_categories = 0                     # num of categories in the category file
        self.category_list = []                     # list of categories
        self.category_index_dict = {}               # unique index for each category

        self.num_targets = 0                        # num unique targets in category file
        self.target_list = []                       # list of unique targets
        self.target_index_dict = {}                 # unique index for each target

        self.target_category_dict = {}              # dictionary of targets pointing to their category deer: herbivore
        self.category_target_list_dict = {}         # dictionary of categories pointing to list of targets in that category herbivore: [deer, elephant]

        self.guess_accuracy_df = None
        self.guess_accuracy_df_list = []
        self.target_balanced_accuracy_df = None

        self.confusion_matrix = np.zeros([self.num_categories, self.num_categories])

        self.threshold_list = np.linspace(-1,1,self.num_thresholds)

        self.load_categories()
        self.create_results_df()
        self.create_average_similarity_score_matrix()
        start_time = time.time()
        self.compute_target_balanced_accuracy()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time for computing target_balanced_accuracy for cohyponym task is :', elapsed_time, 'seconds')


    def load_categories(self, from_file = True):
        if from_file:
            f = open(self.category_info)
            for line in f:
                data = (line.strip().strip('\n').strip()).split(',')
                target = data[0]
                category = data[1]

                if not category in self.category_index_dict:
                    self.category_list.append(category)
                    self.category_index_dict[category] = self.num_categories
                    self.num_categories += 1
                    self.category_target_list_dict[category] = []

                if target in self.target_list:
                    raise Exception("ERROR: Target {} is duplicated in category file".format(target))
                else:
                    self.target_list.append(target)
                    self.target_index_dict[target] = self.num_targets
                    self.num_targets += 1
                    self.target_category_dict[target] = category
                    self.category_target_list_dict[category].append(target)
        else:
            for target, category in self.category_info.items():
                if not category in self.category_index_dict:
                    self.category_list.append(category)
                    self.category_index_dict[category] = self.num_categories
                    self.num_categories += 1
                    self.category_target_list_dict[category] = []

                if target in self.target_list:
                    raise Exception("ERROR: Target {} is duplicated in category file".format(target))
                else:
                    self.target_list.append(target)
                    self.target_index_dict[target] = self.num_targets
                    self.num_targets += 1
                    self.target_category_dict[target] = category
                    self.category_target_list_dict[category].append(target)
            

    def create_results_df(self):
        # note it would seem that there is redundancy here. When data_matrix is symmetic (i.e. when data_matrix[i,j]==data_matrix[j,i])
        #   we are producing duplicate rows, as the results for all target1-target2 comparisons are identical to target2-target1
        #   similarly, when target1=target2, in many models this is identity and we could compute faster by just assuming the answer
        #   but... the matrix will not always be symmetric, and the diagongal of data_matrix won't always = 1, such as in cases where
        #   the data matrix is a co-occurrence conditional probability like P(target1|target2)

        print("\nTesting Guesses")
        self.guess_accuracy_df = pd.DataFrame(columns=['threshold', 
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
        results = []
        for i in range(self.num_thresholds):
            threshold = self.threshold_list[i]
            for j in range(self.num_targets):
                target1 = self.target_list[j]
                category1 = self.target_category_dict[target1]
                    
                for k in range(self.num_targets):
                    target2 = self.target_list[k]
                    category2 = self.target_category_dict[target2]
                    
                    if category1 == category2:
                        actual = 1
                    else:
                        actual = 0
                    target1_index = 'None'
                    target2_index = 'None'

                    if target1 not in self.data_index_dict:
                        target1_index = 'NA'

                    if target2 not in self.data_index_dict:
                        target2_index = 'NA'

                    if target1_index == 'NA' or target2_index == 'NA':
                            guess = 'NA'
                            correct = 'NA'
                            sd_result = 'NA'
                    else:
                        target1_index = self.data_index_dict[target1]
                        target2_index = self.data_index_dict[target2]

                        score = round(self.data_matrix[self.data_index_dict[target1], self.data_index_dict[target2]],10)
                        if 1 < score < 0:
                            raise Exception("ERROR: score must be between 0 and 1")

                        if score >= threshold:
                            guess = 1
                        else:
                            guess = 0

                        if guess == actual:
                            correct = 1
                            if guess == 1:
                                sd_result = 'hit'
                            else:
                                sd_result = 'correct_rejection'
                        else:
                            correct = 0
                            if guess == 1:
                                sd_result = 'false_alarm'
                            else:
                                sd_result = 'miss'

                        result = [threshold, 
                                target1,
                                target2, 
                                category1, 
                                category2, 
                                target1_index, 
                                target2_index, 
                                score, 
                                guess, 
                                actual, 
                                correct, 
                                sd_result]

                        results.append(result)
        self.guess_accuracy_df = self.guess_accuracy_df.append(pd.DataFrame(results, columns = self.guess_accuracy_df.columns), ignore_index=True)

    def create_average_similarity_score_matrix(self):
        self.ave_sim_matrix = np.zeros((self.num_categories, self.num_categories))
        df = self.guess_accuracy_df.loc[(self.guess_accuracy_df['threshold'] == self.threshold_list[0]), ["target1", "target2", "category1", "category2", "score"]]
        df = df.reset_index(drop=True)
        for category1, index1 in self.category_index_dict.items():
            for category2, index2 in self.category_index_dict.items():
                    indices = df.index[(df['category1'] == category1) & (df['category2'] == category2) & (df["target1"] != df["target2"])]
                    if indices.tolist():
                        temp_similarity_scores = df.loc[indices]['score'].tolist()
                        similarity_scores = list(set(temp_similarity_scores))
                        ave_sim_score = mean(similarity_scores)
                        self.ave_sim_matrix[index1, index2] = ave_sim_score

    def compute_target_balanced_accuracy(self, exclude_identity_pairs=True):

        # TODO finish the python implimentation below 
        # and then also impliment this as a more direct pandas expression so that we can compare the speeds
        target_balanced_accuracy_matrix = np.zeros([self.num_thresholds, self.num_targets])

        self.target_balanced_accuracy_df = pd.DataFrame(columns=['target', 
                                                                 'category',
                                                                 'category_index',
                                                                 'best_target_threshold',
                                                                 'best_target_threshold_ba',
                                                                 'best_overall_threshold',
                                                                 'best_overall_threshold_ba'])

        target_hit_matrix = np.zeros([self.num_thresholds, self.num_targets])
        target_yes_correct_count_matrix = np.zeros([self.num_thresholds, self.num_targets])

        target_correct_rejection_matrix = np.zeros([self.num_thresholds, self.num_targets])
        target_no_correct_count_matrix = np.zeros([self.num_thresholds, self.num_targets])

        confusion_matrix_list = []

        for i in range(self.num_thresholds):
            confusion_matrix = np.zeros([self.num_categories, self.num_categories])
            for j in range(self.num_targets):
                for k in range(self.num_targets):

                    if j==k and not exclude_identity_pairs:
                        compute_correctness = True
                    elif j!=k:
                        compute_correctness = True
                    else:
                        compute_correctness = False
                    
                    if compute_correctness:
                        target1 = self.target_list[j]
                        target2 = self.target_list[k]

                        index = self.guess_accuracy_df.index[(self.guess_accuracy_df['threshold'] == self.threshold_list[i]) & (self.guess_accuracy_df['target1'] == target1) & (self.guess_accuracy_df['target2'] == target2)]
                        actual = self.guess_accuracy_df.iloc[index]['actual'].tolist()[0]
                        guess = self.guess_accuracy_df.iloc[index]['guess'].tolist()[0]
                        
                        if guess == 1:
                            confusion_matrix[self.category_index_dict[self.target_category_dict[self.target_list[j]]],self.category_index_dict[self.target_category_dict[self.target_list[k]]]] += 1
                        # this wont work because actual and guess are not defined. 
                        # Need to replace with lookup of pandas table for values of actual and guess for this threshold,target1,target2 triplet
                        if actual == 1:
                            target_yes_correct_count_matrix[i,j] += 1 
                            if guess == 1:
                                target_hit_matrix[i,j] += 1
                        else:
                            target_no_correct_count_matrix[i,j] += 1
                            if guess == 0:
                                target_correct_rejection_matrix[i,j] += 1
            confusion_matrix_list.append(confusion_matrix)


        # need to check for divide by zero errors (cases where target_yes_correct_matrix or target_no_correct_matrix are 0, and have 'NaN' in those cells
        hit_rate = (target_hit_matrix/target_yes_correct_count_matrix)
        hit_rate[np.isnan(hit_rate)] = 0
        correct_rejection_rate = target_correct_rejection_matrix/target_no_correct_count_matrix
        correct_rejection_rate[np.isnan(correct_rejection_rate)] = 0
        #print(self.guess_accuracy_df)
        # this used to be single dimensional, but i made it thresholds,num_targets, and so the code on line 200-214 might not work anymore

        target_balanced_accuracy_matrix = (hit_rate + correct_rejection_rate) / 2
            
        # we want this to be each word's best score across all of its thresholds
        best_overall_scores = np.amax(target_balanced_accuracy_matrix, 0)  #### if we want every column, should the second argument be a 0 or a 1?
        best_overall_score_index = np.argmax(target_balanced_accuracy_matrix, 0)

        # figure out which threshold performs the best overall
        threshold_means = target_balanced_accuracy_matrix.mean(1)
        best_threshold_index = np.argmax(threshold_means)
        self.confusion_matrix = confusion_matrix_list[best_threshold_index]
        best_overall_threshold = self.threshold_list[best_threshold_index]
        best_overall_threshold_scores = target_balanced_accuracy_matrix[best_threshold_index, :]

        results = []
        for i in range(self.num_targets):
            result = [self.target_list[i],
                      self.target_category_dict[self.target_list[i]],
                      self.category_index_dict[self.target_category_dict[self.target_list[i]]],
                      self.threshold_list[best_overall_score_index[i]],
                      best_overall_scores[i],
                      best_overall_threshold,
                      best_overall_threshold_scores[i]]
            results.append(result)
        self.target_balanced_accuracy_df = self.target_balanced_accuracy_df.append(pd.DataFrame(results,columns=self.target_balanced_accuracy_df.columns), ignore_index=True)
        self.guess_accuracy_df_with_best_threshold = self.guess_accuracy_df.loc[self.guess_accuracy_df['threshold'] == best_overall_threshold]




    def compute_category_balanced_accuracy(self, ignore_identity_pairs=True):
        # TODO finish the python implimentation below 
        # and then also impliment this as a more direct pandas expression so that we can compare the speeds
        target_balanced_accuracy_matrix = np.zeros([self.num_thresholds, self.num_targets])

        target_hit_matrix = np.zeros([self.num_thresholds])
        target_yes_correct_count_matrix = np.zeros([self.num_thresholds])

        target_correct_rejection_matrix = np.zeros([self.num_thresholds])
        target_no_correct_count_matrix = np.zeros([self.num_thresholds])