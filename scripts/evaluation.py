from typing import Optional
import sys

sys.path.append("/Users/jingfengzhang/FirstYearProject/SemanticModels2")
from pathlib import Path
from ludwig.results import gen_param_paths
from distributionalmodels.params import param2default, param2requests
from distributionalmodels.models.srn import SRN
from distributionalmodels.models.lstm import LSTM
from distributionalmodels.models.word2vec import W2Vec
from distributionalmodels.models.gpt import GPT
from scripts.evaluation_tools import *
from scripts.utils import *
from scripts.visualization import *
import os
import pandas as pd

np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=np.inf)


class Evaluation:
    def __init__(self, average_over_models=False):
        model_type = 'srn'
        LUDWIG_DATA_PATH: Optional[Path] = None
        RUNS_PATH = Path('/Users/jingfengzhang/FirstYearProject/DistributionalModels/runs')
        self.export_path = Path(
            '/Users/jingfengzhang/FirstYearProject/DistributionalModels/results/{}'.format(model_type))
        self.export_path.mkdir(exist_ok=True)
        # config.Dirs.runs if loading runs locally or None if loading data from ludwig

        project_name = "DistributionalModels"

        dsms_syntagmatictask_using_weights_df_list = []
        self.dsms_syntagmatictask_using_sum_outputs_df_list = []
        self.dsms_syntagmatictask_using_mean_outputs_df_list = []
        self.dsms_cohyponymtask_using_weights_df_list = []
        self.dsms_cohyponymtask_using_hidden_df_list = []
        self.dsms_weight_similarity_matrix_list = []
        self.dsms_hidden_similarity_matrix_list = []
        self.dsms_output_similarity_matrix_list = []
        self.category_colnames = {'0': 'Period', '1': 'Present A', '2': 'Omitted A', '3': 'Legal As', '4': 'Illegal As',
                                  '5': 'Present B', '6': 'Omitted B', '7': 'Legal Bs', '8': 'Illegal Bs', '9': 'y'}
        self.dsm_list = []
        i = 0
        self.current_dir = ''
        for param_path, label in gen_param_paths(project_name,
                                                 param2requests,
                                                 param2default,
                                                 isolated=True if RUNS_PATH is not None else False,
                                                 runs_path=RUNS_PATH,
                                                 ludwig_data_path=LUDWIG_DATA_PATH,
                                                 require_all_found=False,
                                                 ):
            # load models and embeddings
            dsm_weight_list = []
            dsm_hidden_activation_list = []
            dsm_output_activation_list = []
            # dsm_info = [dsm.params.num_epochs, dsm.params.learning_rate, dsm.params.embed_size]

            if model_type == 'w2v':
                self.dsm_list, self.checkpoint_list, corpus = W2Vec.from_pretrained(param_path)
                self.dsm_info = [self.dsm_list[0].params.num_epochs, self.dsm_list[0].params.learning_rate,
                                 self.dsm_list[0].params.embed_size,
                                 self.dsm_list[0].params.embed_init_range, self.dsm_list[0].params.momentum,
                                 self.dsm_list[0].params.round]
                for w2v in self.dsm_list:
                    dsm_weight_list.append(w2v.model.embed.weight.clone().detach().numpy())
                    w2v_outputs, w2v_hiddens = get_representations_from_w2v(w2v)
                    dsm_output_activation_list.append(w2v_outputs)
                    dsm_hidden_activation_list.append(w2v_hiddens)
            elif model_type == 'srn':
                self.dsm_list, self.checkpoint_list, corpus = SRN.from_pretrained(param_path)
                self.dsm_info = [self.dsm_list[0].params.num_epochs, self.dsm_list[0].params.learning_rate,
                                 self.dsm_list[0].params.embed_size,
                                 self.dsm_list[0].params.embed_init_range, self.dsm_list[0].params.momentum,
                                 self.dsm_list[0].params.round]
                for srn in self.dsm_list:
                    srn_outputs, srn_hiddens = get_representations_from_srn(srn)
                    dsm_output_activation_list.append(srn_outputs)
                    dsm_hidden_activation_list.append(srn_hiddens)
                    dsm_weight_list.append(srn.model.h_x.weight.clone().detach().numpy().transpose(1, 0))
            elif model_type == 'lstm':
                self.dsm_list, self.checkpoint_list, corpus = LSTM.from_pretrained(param_path)
                self.dsm_info = [self.dsm_list[0].params.num_epochs, self.dsm_list[0].params.learning_rate,
                                 self.dsm_list[0].params.embed_size,
                                 self.dsm_list[0].params.embed_init_range, self.dsm_list[0].params.momentum,
                                 self.dsm_list[0].params.round]
                for lstm in self.dsm_list:
                    dsm_weight_list.append(lstm.model.wx.weight.clone().detach().numpy())
                    lstm_outputs, lstm_hiddens = get_representations_from_lstm(lstm)
                    dsm_output_activation_list.append(lstm_outputs)
                    dsm_hidden_activation_list.append(lstm_hiddens)
            elif model_type == 'gpt':
                self.dsm_list, self.checkpoint_list, corpus = GPT.from_pretrained(param_path)
                self.dsm_info = [self.dsm_list[0].params.num_epochs, self.dsm_list[0].params.learning_rate,
                                 self.dsm_list[0].params.embed_size, self.dsm_list[0].params.block_size,
                                 self.dsm_list[0].params.head_size, self.dsm_list[0].params.round]
                for gpt in self.dsm_list:
                    gpt_outputs, gpt_multihead_outputs, gpt_weights = get_representations_from_gpt(gpt)
                    dsm_output_activation_list.append(gpt_outputs)
                    dsm_hidden_activation_list.append(gpt_multihead_outputs)
                    dsm_weight_list.append(gpt_weights)

            self.current_dir = str(self.export_path) + '/{}_{}_{}_{}_{}_{}'.format(self.dsm_info[0], self.dsm_info[1],
                                                                                   self.dsm_info[2], self.dsm_info[3],
                                                                                   self.dsm_info[4], self.dsm_info[5])
            os.makedirs(self.current_dir, exist_ok=True)

            dsm_weight_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'), dsm_weight_list))
            self.dsms_weight_similarity_matrix_list.append(dsm_weight_similarity_matrix_list)

            dsm_hidden_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'),
                    dsm_hidden_activation_list))
            self.dsms_hidden_similarity_matrix_list.append(dsm_hidden_similarity_matrix_list)

            dsm_output_similarity_matrix_list = list(
                map(lambda embed_matrix: create_similarity_matrix(embed_matrix, 'corrcoef'),
                    dsm_output_activation_list))
            self.dsms_output_similarity_matrix_list.append(dsm_hidden_similarity_matrix_list)

            self.combine_and_export_data(dsm_weight_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_weights_from_different_checkpoints.csv')
            self.combine_and_export_data(dsm_hidden_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_hidden_from_different_checkpoints.csv')
            self.combine_and_export_data(dsm_output_activation_list, self.checkpoint_list,
                                         self.current_dir + '/combined_outputs_from_different_checkpoints.csv')
            self.combine_and_export_data(dsm_output_similarity_matrix_list, self.checkpoint_list,
                                         self.current_dir + '/combined_outputs_sim_from_different_checkpoints.csv')


            dsm_cohyponymtask_using_weights_list = [[] for x in range(3)]
            dsm_cohyponymtask_using_hidden_activation_list = [[] for x in range(3)]
            dsm_cohyponymtask_using_output_activation_list = [[] for x in range(3)]
            dsm_syntagmatictask_using_weights_list = [[] for x in range(3)]
            dsm_syntagmatictask_using_sum_outputs_list = [[] for x in range(3)]
            dsm_syntagmatictask_using_mean_outputs_list = [[] for x in range(3)]
            for i in range(len(self.checkpoint_list)):
                cohyponymtask_using_weights = CohyponymTask(corpus.paradigmatic_word_category_dict,
                                                            dsm_weight_similarity_matrix_list[i],
                                                            self.dsm_list[i].vocab_id_dict)
                mean_acivtation_weights, sum_acivtation_weights, guess_accuracies_weights = evaluate_srn(
                    self.checkpoint_list[i], self.dsm_list[i], cohyponymtask_using_weights)
                cohyponymtask_using_hidden = CohyponymTask(corpus.paradigmatic_word_category_dict,
                                                           dsm_hidden_similarity_matrix_list[i], dsm.vocab_dict)
                mean_acivtation_weights, sum_acivtation_weights, guess_accuracies_weights = evaluate_srn(
                    self.checkpoint_list[i], self.dsm_list[i], cohyponymtask_using_hidden)

                mean_acivtation_weights, sum_acivtation_weights, guess_accuracies_weights = evaluate_model(
                    self.dsm_list[i], model_type, self.checkpoint_list[i], corpus.numeric_token_sequence_list,
                    corpus.sequence_test_category_list, dsm_weight_similarity_matrix_list[i],
                    cohyponymtask_using_weights)
                mean_acivtation_hidden, sum_acivtation_hidden, guess_accuracies_hidden = evaluate_model(
                    self.dsm_list[i], model_type, self.checkpoint_list[i], corpus.numeric_token_sequence_list,
                    corpus.sequence_test_category_list, dsm_hidden_similarity_matrix_list[i],
                    cohyponymtask_using_hidden)
                mean_acivtation_outputs, sum_acivtation_outputs, guess_accuracies_outputs = evaluate_model(
                    self.dsm_list[i], model_type, self.checkpoint_list[i], corpus.numeric_token_sequence_list,
                    corpus.sequence_test_category_list)
                for j in range(3):
                    # dsm_syntagmatictask_using_weights_list[j].append(mean_acivtation_weights[j])
                    dsm_syntagmatictask_using_sum_outputs_list[j].append(sum_acivtation_outputs[j])
                    dsm_syntagmatictask_using_mean_outputs_list[j].append(mean_acivtation_outputs[j])
                    dsm_cohyponymtask_using_weights_list[j].append(guess_accuracies_weights[j])
                    dsm_cohyponymtask_using_hidden_activation_list[j].append(guess_accuracies_hidden[j])
                if i == len(self.checkpoint_list) - 1:
                    cohyponymtask_using_weights.guess_accuracy_best_df.to_csv(
                        self.current_dir + '/weight_guess_accuracy.csv')
                    cohyponymtask_using_hidden.guess_accuracy_best_df.to_csv(
                        self.current_dir + '/hidden_guess_accuracy.csv')
            temp_dsms_cohyponymtask_using_weights_df = self.merge_data_from_checkpoints(
                dsm_cohyponymtask_using_weights_list, self.dsms_cohyponymtask_using_weights_df_list)
            temp_dsms_cohyponymtask_using_hidden_df = self.merge_data_from_checkpoints(
                dsm_cohyponymtask_using_hidden_activation_list, self.dsms_cohyponymtask_using_hidden_df_list)
            temp_dsms_syntagmatictask_using_sum_outputs_df = self.merge_data_from_checkpoints(
                dsm_syntagmatictask_using_sum_outputs_list, self.dsms_syntagmatictask_using_sum_outputs_df_list)
            temp_dsms_syntagmatictask_using_mean_outputs_df = self.merge_data_from_checkpoints(
                dsm_syntagmatictask_using_mean_outputs_list, self.dsms_syntagmatictask_using_mean_outputs_df_list)

            plot_prediction_accuracy(temp_dsms_syntagmatictask_using_sum_outputs_df,
                                     temp_dsms_syntagmatictask_using_mean_outputs_df,
                                     self.current_dir + '/sum_prediction_activation_outputs_plot.png')
            plot_cohyponym_score(temp_dsms_cohyponymtask_using_weights_df, temp_dsms_cohyponymtask_using_hidden_df,
                                 self.current_dir + '/cohyponym_plot.png')

            i += 1

    def combine_and_export_data(self, data_list, keys, dir):
        vocab_list = list(self.dsm_list[0].vocab_id_dict.keys())
        dfs = [pd.DataFrame(np.round(data, decimals=3), columns=vocab_list, index=vocab_list) for data in data_list]
        export_df = pd.concat(dfs, keys=keys)
        export_path = export_df.to_csv(Path(dir))

    def merge_data_from_checkpoints(self, data_list, dsms_list):
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

        dsms_list.append(temp_df)
        return temp_df

    def average_over_multiple_models(self, dsms_list):
        temp_df = dsms_list[0].copy()
        dsm_df = pd.concat(dsms_list)
        dsm_df = dsm_df.groupby(dsm_df.index).mean()
        dsm_df = dsm_df.rename(columns=self.category_colnames)
        dsm_df['checkpoint'] = temp_df['checkpoint']
        dsm_df['input'] = temp_df['input']
        return dsm_df

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
        # syntagmatictask_using_weights_df = self.average_over_multiple_models(dsms_syntagmatictask_using_weights_df_list, self.merge_data_from_checkpoints(dsm_syntagmatictask_using_weights_list,dsms_syntagmatictask_using_weights_df_list))
        cohyponymtask_using_weights_df = self.average_over_multiple_models(
            self.dsms_cohyponymtask_using_weights_df_list)
        cohyponymtask_using_hidden_df = self.average_over_multiple_models(self.dsms_cohyponymtask_using_hidden_df_list)
        syntagmatictask_using_sum_outputs_df = self.average_over_multiple_models(
            self.dsms_syntagmatictask_using_sum_outputs_df_list)
        syntagmatictask_using_mean_outputs_df = self.average_over_multiple_models(
            self.dsms_syntagmatictask_using_mean_outputs_df_list)
        plot_prediction_accuracy(syntagmatictask_using_sum_outputs_df,
                                 syntagmatictask_using_mean_outputs_df,
                                 average_dir + '/sum_prediction_activation_outputs_plot.png')
        plot_cohyponym_score(cohyponymtask_using_weights_df, cohyponymtask_using_hidden_df,
                             average_dir + '/cohyponym_average_plot.png')
        plot_similarity_matrices(self.average_lists_of_matrices(self.dsms_weight_similarity_matrix_list), 'weight',
                                 self.checkpoint_list, self.words, self.groups, average_dir)
        plot_similarity_matrices(self.average_lists_of_matrices(self.dsms_hidden_similarity_matrix_list), 'hidden',
                                 self.checkpoint_list, self.words, self.groups, average_dir)


eval = Evaluation()
eval.average_results()
