import pandas as pd
import scipy.stats as st
from statistics import mean

class Result:

    def __init__(self):

        self.results = None
        self.model = None

    def get_results(self, target_balanced_accuracy_df, model_type):
        self.results = []
            # lets' say we want to save the mean and std of the matrix (hypothetically)
        if 'best_overall_threshold_ba' in target_balanced_accuracy_df:
            mean = target_balanced_accuracy_df['best_overall_threshold_ba'].mean()
            std = target_balanced_accuracy_df['best_overall_threshold_ba'].std()
            sem = target_balanced_accuracy_df['best_overall_threshold_ba'].sem()
            ci95 = mean - st.t.interval(alpha=0.95, df=target_balanced_accuracy_df.shape[0]-1, loc=mean, scale=sem)[0]
            best_overall_threshold_ba_series = pd.Series({'model_type' : model_type,'sample_size': target_balanced_accuracy_df.shape[0], 'mean' : mean, 'std' : std, 'sem' : sem, 'ci95': ci95})
            best_overall_threshold_ba_series.name = 'total_summary'
            self.results.append(best_overall_threshold_ba_series)

        if 'category' in target_balanced_accuracy_df:
            print('category' in target_balanced_accuracy_df)
            category_ba_dict = {}
            for category in target_balanced_accuracy_df.category.unique():
                num_target_in_category = len(target_balanced_accuracy_df.loc[target_balanced_accuracy_df['category'] == category])
                mean = target_balanced_accuracy_df.loc[target_balanced_accuracy_df['category'] == category, 'best_target_threshold_ba'].mean()
                category_ba_dict[category + '_MEAN'] = mean
                std = target_balanced_accuracy_df.loc[target_balanced_accuracy_df['category'] == category, 'best_target_threshold_ba'].std()
                sem = target_balanced_accuracy_df.loc[target_balanced_accuracy_df['category'] == category, 'best_target_threshold_ba'].sem()
                category_ba_dict[category + '_CI95'] = mean - st.t.interval(alpha=0.95, df=num_target_in_category-1, loc=mean, scale=sem)[0]
                category_ba_series = pd.Series(category_ba_dict)
                category_ba_series.name = 'category_summary'
            self.results.append(category_ba_series)
        else:
            print('warning! no data found')

    def get_empty_result(self):
        self.results = []
        result = pd.Series()
        result.name = 'none'
        self.results.append(result)
        # self.results = []  # each results produces a single csv file
        # for result_value, result_name in zip([mean_of_wd, std_of_wd], ['mean', 'std']):
        #     # convert result to pandas series
        #     result = pd.Series([result_value, result_value], index=['before-training', 'after_training'])
        #     result.name = result_name  # name of the csv file
        #     self.results.append(result)

    # def get_results(self, data_matrix):
    #         # lets' say we want to save the mean and std of the matrix (hypothetically)
    #     mean_of_wd = data_matrix.mean().mean()
    #     std_of_wd = data_matrix.std().mean()

    #     self.results = []  # each results produces a single csv file
    #     for result_value, result_name in zip([mean_of_wd, std_of_wd], ['mean', 'std']):
    #         # convert result to pandas series
    #         result = pd.Series([result_value, result_value], index=['before-training', 'after_training'])
    #         result.name = result_name  # name of the csv file
    #         self.results.append(result)    
