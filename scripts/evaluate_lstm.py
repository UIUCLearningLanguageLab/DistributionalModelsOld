import sys

sys.path.append("/Users/jingfengzhang/FirstYearProject/SemanticModels2")
from pathlib import Path
from ludwig.results import gen_param_paths
from distributionalmodels.params import param2default, param2requests
from scripts.evaluation_tools import *
from scripts.utils import *
from scripts.visualization import *
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=np.inf)


class LSTMEvaluation:
    def __init__(self, average_over_models=False):
        LUDWIG_DATA_PATH: Optional[Path] = None
        # RUNS_PATH = Path('/Users/jingfengzhang/FirstYearProject/DistributionalModels/runs')
        RUNS_PATH = Path('/Volumes/ludwig_data/DistributionalModels/runs')
        self.export_path = Path(
            '/Users/jingfengzhang/FirstYearProject/DistributionalModels/results/lstm')
        os.makedirs(self.export_path, exist_ok=True)
        self.export_path.mkdir(exist_ok=True)

        self.lstms_syntagmatictask_using_sum_outputs_df_list = []
        self.lstms_syntagmatictask_using_mean_outputs_df_list = []
        self.lstms_syntagmatictask_using_weights_df_list = []
        self.lstms_syntagmatictask_using_hidden_df_list = []
        self.lstms_cohyponymtask_using_weights_df_list = []
        self.lstms_cohyponymtask_using_hidden_df_list = []
        self.lstms_cohyponymtask_using_outputs_df_list = []
        self.lstms_sim_score_using_weights_df_list = []
        self.lstms_sim_score_using_hidden_df_list = []
        self.lstms_sim_score_using_outputs_df_list = []

        self.lstms_weight_similarity_matrix_list = []
        self.lstms_hidden_similarity_matrix_list = []
        self.lstms_output_similarity_matrix_list = []

        self.category_colnames = {'0': 'Period', '1': 'Present A', '2': 'Omitted A', '3': 'Legal As', '4': 'Illegal As',
                                  '5': 'Present B', '6': 'Omitted B', '7': 'Legal Bs', '8': 'Illegal Bs', '9': 'y'}
        self.lstm_list = []
        i = 0
        self.current_dir = ''
        for param_path, label in gen_param_paths("DistributionalModels",
                                                 param2requests,
                                                 param2default,
                                                 isolated=True if RUNS_PATH is not None else False,
                                                 runs_path=RUNS_PATH,
                                                 ludwig_data_path=LUDWIG_DATA_PATH,
                                                 require_all_found=False,
                                                 ):
            # load models and embeddings
            lstm_weight_list = []
            lstm_hidden_activation_list = []
            lstm_output_activation_list = []

            self.lstm_list, self.checkpoint_list, corpus = LSTM.from_pretrained(param_path)
            self.lstm_info = [self.lstm_list[0].params.num_epochs, self.lstm_list[0].params.learning_rate,
                              self.lstm_list[0].params.embed_size,
                              self.lstm_list[0].params.embed_init_range, self.lstm_list[0].params.momentum,
                              self.lstm_list[0].params.round]
            for lstm in self.lstm_list:
                lstm_outputs, lstm_hiddens = get_representations_from_lstm(lstm)
                lstm_output_activation_list.append(lstm_outputs)
                lstm_hidden_activation_list.append(lstm_hiddens)
                lstm_weight_list.append(lstm.model.wx.weight.clone().detach().numpy())

            self.current_dir = str(self.export_path) + '/{}_{}_{}_{}_{}_{}'.format(self.lstm_info[0], self.lstm_info[1],
                                                                                   self.lstm_info[2], self.lstm_info[3],
                                                                                   self.lstm_info[4], self.lstm_info[5])
            os.makedirs(self.current_dir, exist_ok=True)

            lstm_weight_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'), lstm_weight_list))
            self.lstms_weight_similarity_matrix_list.append(lstm_weight_similarity_matrix_list)

            lstm_hidden_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'),
                    lstm_hidden_activation_list))
            self.lstms_hidden_similarity_matrix_list.append(lstm_hidden_similarity_matrix_list)

            lstm_output_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'),
                    lstm_output_activation_list))
            self.lstms_output_similarity_matrix_list.append(lstm_hidden_similarity_matrix_list)

            self.combine_and_export_data(lstm_weight_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_weights_from_different_checkpoints.csv')
            self.combine_and_export_data(lstm_hidden_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_hidden_from_different_checkpoints.csv')
            self.combine_and_export_data(lstm_output_activation_list, self.checkpoint_list,
                                         self.current_dir + '/combined_outputs_from_different_checkpoints.csv')
            self.combine_and_export_data(lstm_output_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_outputs_sim_from_different_checkpoints.csv')

            vocab_list = list(self.lstm_list[0].vocab_id_dict.keys())
            hidden_sim_df = pd.DataFrame(np.round(lstm_hidden_similarity_matrix_list[-1], decimals=3),
                                         columns=vocab_list, index=vocab_list)

            lstm_cohyponymtask_using_weights_list = [[] for x in range(3)]
            lstm_cohyponymtask_using_hidden_activation_list = [[] for x in range(3)]
            lstm_cohyponymtask_using_output_activation_list = [[] for x in range(3)]
            lstm_sim_score_using_weights_list = [[] for x in range(3)]
            lstm_sim_score_using_hidden_activation_list = [[] for x in range(3)]
            lstm_sim_score_using_output_activation_list = [[] for x in range(3)]
            lstm_syntagmatictask_using_weights_list = [[] for x in range(3)]
            lstm_syntagmatictask_using_hidden_list = [[] for x in range(3)]
            lstm_syntagmatictask_using_sum_outputs_list = [[] for x in range(3)]
            lstm_syntagmatictask_using_mean_outputs_list = [[] for x in range(3)]
            for i in tqdm(range(len(self.checkpoint_list))):
                cohyponymtask_using_weights = CohyponymTask(corpus.paradigmatic_word_category_dict,
                                                            lstm_weight_similarity_matrix_list[i],
                                                            self.lstm_list[i].vocab_id_dict)
                mean_acivtation_weights, sum_acivtation_weights, guess_accuracies_weights, sim_scores_weights = evaluate_lstm(
                    self.checkpoint_list[i], self.lstm_list[i], cohyponymtask_using_weights)

                cohyponymtask_using_hidden = CohyponymTask(corpus.paradigmatic_word_category_dict,
                                                           lstm_hidden_similarity_matrix_list[i],
                                                           self.lstm_list[i].vocab_id_dict)
                mean_acivtation_hidden, sum_acivtation_hidden, guess_accuracies_hidden, sim_scores_hidden = evaluate_lstm(
                    self.checkpoint_list[i], self.lstm_list[i], cohyponymtask_using_hidden)

                cohyponymtask_using_outputs = CohyponymTask(corpus.paradigmatic_word_category_dict,
                                                            lstm_output_similarity_matrix_list[i],
                                                            self.lstm_list[i].vocab_id_dict)
                mean_acivtation_outputs, sum_acivtation_outputs, guess_accuracies_outputs, sim_scores_outputs = evaluate_lstm(
                    self.checkpoint_list[i], self.lstm_list[i], cohyponymtask_using_outputs)

                for j in range(3):
                    lstm_syntagmatictask_using_weights_list[j].append(mean_acivtation_weights[j])
                    lstm_syntagmatictask_using_hidden_list[j].append(mean_acivtation_hidden[j])
                    lstm_syntagmatictask_using_sum_outputs_list[j].append(sum_acivtation_outputs[j])
                    lstm_syntagmatictask_using_mean_outputs_list[j].append(mean_acivtation_outputs[j])

                    lstm_cohyponymtask_using_weights_list[j].append(guess_accuracies_weights[j])
                    lstm_cohyponymtask_using_hidden_activation_list[j].append(guess_accuracies_hidden[j])
                    lstm_cohyponymtask_using_output_activation_list[j].append(guess_accuracies_outputs[j])

                    lstm_sim_score_using_weights_list[j].append(sim_scores_weights[j])
                    lstm_sim_score_using_hidden_activation_list[j].append(sim_scores_hidden[j])
                    lstm_sim_score_using_output_activation_list[j].append(sim_scores_outputs[j])


                if i == len(self.checkpoint_list) - 1:
                    cohyponymtask_using_weights.guess_accuracy_best_df.to_csv(
                        self.current_dir + '/weight_guess_accuracy.csv')
                    cohyponymtask_using_hidden.guess_accuracy_best_df.to_csv(
                        self.current_dir + '/hidden_guess_accuracy.csv')
                    cohyponymtask_using_outputs.guess_accuracy_best_df.to_csv(
                        self.current_dir + '/outputs_guess_accuracy.csv')

            temp_lstms_cohyponymtask_using_weights_df = self.merge_data_from_checkpoints(
                lstm_cohyponymtask_using_weights_list, self.lstms_cohyponymtask_using_weights_df_list)
            temp_lstms_cohyponymtask_using_hidden_df = self.merge_data_from_checkpoints(
                lstm_cohyponymtask_using_hidden_activation_list, self.lstms_cohyponymtask_using_hidden_df_list)
            temp_lstms_cohyponymtask_using_outputs_df = self.merge_data_from_checkpoints(
                lstm_cohyponymtask_using_output_activation_list, self.lstms_cohyponymtask_using_outputs_df_list)

            temp_lstms_sim_score_using_hidden_df = self.merge_data_from_checkpoints(
                lstm_sim_score_using_hidden_activation_list, self.lstms_sim_score_using_hidden_df_list)
            temp_lstms_sim_score_using_weights_df = self.merge_data_from_checkpoints(
                lstm_sim_score_using_weights_list, self.lstms_sim_score_using_weights_df_list)
            temp_lstms_sim_score_using_outputs_df = self.merge_data_from_checkpoints(
                lstm_sim_score_using_output_activation_list, self.lstms_sim_score_using_outputs_df_list)

            temp_lstms_syntagmatictask_using_sum_outputs_df = self.merge_data_from_checkpoints(
                lstm_syntagmatictask_using_sum_outputs_list, self.lstms_syntagmatictask_using_sum_outputs_df_list)
            temp_lstms_syntagmatictask_using_mean_outputs_df = self.merge_data_from_checkpoints(
                lstm_syntagmatictask_using_mean_outputs_list, self.lstms_syntagmatictask_using_mean_outputs_df_list)
            temp_lstms_syntagmatictask_using_weights_df = self.merge_data_from_checkpoints(
                lstm_syntagmatictask_using_weights_list, self.lstms_syntagmatictask_using_weights_df_list)
            temp_lstms_syntagmatictask_using_hidden_df = self.merge_data_from_checkpoints(
                lstm_syntagmatictask_using_hidden_list, self.lstms_syntagmatictask_using_hidden_df_list)

            plot_prediction_accuracy(temp_lstms_syntagmatictask_using_sum_outputs_df,
                                     temp_lstms_syntagmatictask_using_mean_outputs_df,
                                     self.current_dir + '/sum_prediction_activation_outputs_plot.png')
            plot_cohyponym_score(temp_lstms_cohyponymtask_using_weights_df, temp_lstms_cohyponymtask_using_hidden_df,
                                 self.current_dir + '/cohyponym_plot.png')
            plot_sim_score(temp_lstms_sim_score_using_hidden_df, 'hidden',  self.current_dir + '/sim_score_plot.png')
            plot_confusion_matrix(hidden_sim_df, self.current_dir + '/confusion_matrix_plot.png')

            i += 1

    def combine_and_export_data(self, data_list, keys, dir):
        vocab_list = list(self.lstm_list[0].vocab_id_dict.keys())
        dfs = [pd.DataFrame(np.round(data, decimals=3), columns=vocab_list, index=vocab_list) for data in data_list]
        export_df = pd.concat(dfs, keys=keys)
        export_path = export_df.to_csv(Path(dir))

    def merge_data_from_checkpoints(self, data_list, lstms_list):
        temp_df_list = [pd.DataFrame.from_dict(data_list[i]) for i in range(len(data_list))]
        for k in range(len(data_list)):
            if k == 0:
                temp_df_list[k]['input'] = 'A'
            elif k == 1:
                temp_df_list[k]['input'] = 'y'
            else:
                temp_df_list[k]['input'] = 'B'

        temp_df = pd.concat(temp_df_list, ignore_index=True)
        temp_df = temp_df.rename(columns=self.category_colnames)

        lstms_list.append(temp_df)
        return temp_df

    def average_over_multiple_models(self, lstms_list):
        temp_df = lstms_list[0].copy()
        lstm_df = pd.concat(lstms_list)
        lstm_df = lstm_df.drop('checkpoint', axis=1)
        lstm_df = lstm_df.drop('input', axis=1)
        lstm_df = lstm_df.groupby(lstm_df.index).mean()
        lstm_df = lstm_df.rename(columns=self.category_colnames)
        lstm_df['checkpoint'] = temp_df['checkpoint']
        lstm_df['input'] = temp_df['input']
        return lstm_df

    def average_lists_of_matrices(self, list_of_lists_of_matrices):
        # Get the dimensions of the matrices
        num_lists = len(list_of_lists_of_matrices)
        num_matrices = len(list_of_lists_of_matrices[0])

        # Initialize an empty list to store the averaged matrices
        averaged_matrices = []

        # Loop through each matrix
        for i in range(num_matrices):
            # Sum the matrices at the same position in each list
            sum_of_matrices = np.zeros_like(list_of_lists_of_matrices[0][i])
            for j in range(num_lists):
                sum_of_matrices += list_of_lists_of_matrices[j][i]

            # Divide the sum by the number of lists to get the average
            average_matrix = sum_of_matrices / num_lists

            # Add the averaged matrix to the list of averaged matrices
            averaged_matrices.append(average_matrix)

        return averaged_matrices

    def average_results(self):
        average_dir = self.current_dir + '_average'
        os.makedirs(average_dir, exist_ok=True)
        # syntagmatictask_using_weights_df = self.average_over_multiple_models(lstms_syntagmatictask_using_weights_df_list, self.merge_data_from_checkpoints(lstm_syntagmatictask_using_weights_list,lstms_syntagmatictask_using_weights_df_list))
        cohyponymtask_using_weights_df = self.average_over_multiple_models(
            self.lstms_cohyponymtask_using_weights_df_list)
        cohyponymtask_using_hidden_df = self.average_over_multiple_models(self.lstms_cohyponymtask_using_hidden_df_list)
        sim_score_using_hidden_df = self.average_over_multiple_models(self.lstms_sim_score_using_hidden_df_list)
        syntagmatictask_using_sum_outputs_df = self.average_over_multiple_models(
            self.lstms_syntagmatictask_using_sum_outputs_df_list)
        syntagmatictask_using_mean_outputs_df = self.average_over_multiple_models(
            self.lstms_syntagmatictask_using_mean_outputs_df_list)
        plot_prediction_accuracy(syntagmatictask_using_sum_outputs_df,
                                 syntagmatictask_using_mean_outputs_df,
                                 average_dir + '/sum_prediction_activation_outputs_plot.png')
        plot_cohyponym_score(cohyponymtask_using_weights_df, cohyponymtask_using_hidden_df,
                             average_dir + '/cohyponym_average_plot.png')
        plot_sim_score(sim_score_using_hidden_df, 'hidden', self.current_dir + '/sim_score_plot.png')


eval = LSTMEvaluation()
eval.average_results()
