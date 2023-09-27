import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict, OrderedDict
from semanticmodels2.tasks.cohyponym_task import CohyponymTask
import pandas as pd
import torch

# stack the output activation for each unique token in corpus
def get_output_activation(dsm, dsm_type):
    dsm.model.eval()
    if dsm_type == 'srn':
        old_h = dsm.model.init_hidden_state()
    elif dsm_type == 'lstm':
        old_h, old_c = dsm.model.init_hidden_state_2d()
    # define an empty matrix with size vocab_size*vocab_size
    output_activations = np.zeros((dsm.vocab_size, dsm.vocab_size))
    # iterate through all unique tokens in corpus
    for i in range(dsm.vocab_size):
        if dsm_type == 'srn':
            input_vector = dsm.model.embed[i]
            z_h, h, z_o, o, o_prob = dsm.model.forward_item(input_vector, old_h)
            old_h = h.detach()
        elif dsm_type == 'w2v':
            logits, hidden = dsm.model(input=torch.tensor([i]))
            o_prob = torch.nn.functional.softmax(logits, dim=1)
        else:
            z_o, o_prob, h, c = dsm.model(i, old_h, old_c)
            old_h, old_c = h.detach(), c.detach()
        output_activations[i] = o_prob.cpu().detach().numpy()
    return output_activations


# stack recurrent state activation for each unique token
def get_hidden_activation(dsm, dsm_type):
    dsm.model.eval()
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([dsm.vocab_size, dsm.params.embed_size], float)
    if dsm_type == 'srn':
        old_h = dsm.model.init_hidden_state()
    elif dsm_type == 'lstm':
        old_h, old_c = dsm.model.init_hidden_state_2d()
    
    # dict with each unique token as a key, pointing to a list of each of its hidden states
    #   for each time that token occurs in the corpus
    
    # define an empty matrix with size vocab_size*hidden_size

    # iterate through every token contained in the corpus
    for document in dsm.numeric_sequence_list:
        for i in range(len(document)):
            if dsm_type == "srn":
                input_vector = dsm.model.embed[document[i]]
                z_h, h, z_o, o, o_prob = dsm.model.forward_item(input_vector, old_h)
                old_h = h.detach()
                # store every recurrent activtion of each unique token in a dictionary
                hidden_dict[document[i]].append(old_h.numpy())
            elif dsm_type == 'w2v':
                logits, hidden = dsm.model(input=torch.tensor([document[i]]))
                hidden_dict[document[i]].append(hidden.detach().numpy())
            else:
                z_o, o_prob, h, c = dsm.model(document[i], old_h, old_c)
                old_h, old_c = h.detach(), c.detach()
                hidden_dict[document[i]].append(old_h.numpy())

    # Sort the dictionary based on the keys, so that it has the same order as the the vocab dict
    sorted_hidden_dict = OrderedDict(sorted(hidden_dict.items()))

    # average the activations
    for vocab_index, hidden_list in sorted_hidden_dict.items():
        hidden_matrix[vocab_index] = np.mean(np.array(hidden_list), axis=0)
    return hidden_matrix


def evaluate_model(dsm, dsm_type, checkpoint, seq_num, sequence_test_category_list, similarity_matrix=np.array([]) ,cohyponymtask: CohyponymTask =None):
    dsm.model.eval()
    if dsm_type == 'srn':
        old_h = dsm.model.init_hidden_state()
    elif dsm_type == 'lstm':
        old_h, old_c = dsm.model.init_hidden_state_2d()
    mean_group_activation_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
    # iterate through every document
    for doc_index, document in enumerate(seq_num):
        # set the first sentnece index to 0, so that we can keep track of the number of sentences we tested 
        current_sentence_index = 0
        for i in range(len(document[:-1])):
            if dsm_type == 'srn':
                input_vector = dsm.model.embed[document[i]]
                z_h, h, z_o, o, o_prob = dsm.model.forward_item(input_vector, old_h)
                old_h = h.detach()
            elif dsm_type == 'w2v':
                input=torch.tensor([document[i]])
                logits, hidden = dsm.model(input=torch.tensor([document[i]]))
                o_prob = torch.nn.functional.softmax(logits, dim=1)
            else:
                z_o, o_prob, h, c = dsm.model(document[i], old_h, old_c)
                old_h, old_c = h.detach(), c.detach()
            # check if we want to weight or hidden activation or output activation
            if similarity_matrix.size != 0:
                vector = similarity_matrix[document[i]]
            else:
                vector = o_prob.cpu().detach().numpy()[0]
            # get the type of tokens in current sentence
            current_sequence_test_category_array = np.array(sequence_test_category_list[current_sentence_index])
            # check if we want to perform cohyponymtask
            if cohyponymtask != None:
                target_df = cohyponymtask.guess_accuracy_best_df.loc[cohyponymtask.guess_accuracy_best_df.target1 == dsm.id2token[document[i]]]
            else:
                target_df = pd.DataFrame()
            # if the token we enconter belongs to A category
            if dsm.id2token[document[i]][0] == 'A':
                # category_dict = {list(dsm.id2token.values())[j]: current_sequence_test_category_array[j] for j in range(dsm.vocab_size)}
                # category_label_list = dsm.corpus.category_label_list
                # reverse_dict = {}
                # for key, value in category_dict.items():
                #     if value not in reverse_dict:
                #         reverse_dict[value] = [key]
                #     else:
                #         reverse_dict[value].append(key)

                # # Sort the dictionary by its keys
                # sorted_reverse_dict = {k: reverse_dict[k] for k in sorted(reverse_dict)}
                # labeled_dict = {category_label_list[k]: v for k, v in sorted_reverse_dict.items()}
                # print(f'current sentence: {[dsm.id2token[token] for token in dsm.corpus.numeric_token_sequence_sentence_list[doc_index][current_sentence_index]]}')
                # for value, keys in labeled_dict.items():
                #     print(f"Category: {value}, Tokens: {keys}")
                

                for group_index in np.unique(current_sequence_test_category_array):
                    # print(np.where(current_sequence_test_category_array == group_number))
                    # get the index of tokens in the word_dict
                    indices = np.where(current_sequence_test_category_array == group_index)
                    tokens = list(map(dsm.id2token.get, indices[0].tolist()))
                    # compute the mean and sum output activation 
                    mean_group_activation_dict_list[0][str(group_index)].append(np.mean(vector[indices]))
                    sum_group_activation_dict_list[0][str(group_index)].append(np.sum(vector[indices]))
                    # store cohyponym results
                    if target_df.empty != True:
                        results = target_df.loc[target_df['target2'].isin(tokens)]['sd_result'].to_list()
                        correct_number = 0
                        for result in results:
                            if result == 'hit' or result == 'cr':
                                correct_number +=1
                        guess_accuracy_dict_list[0][str(group_index)].append(correct_number/len(results))
                        # print('results: {}, proportion: {}'.format(results, correct_number/len(results)))
                # print('breakline -----------------------------------')


            elif dsm.id2token[document[i]][0] == 'y':
                # print(current_sequence_test_category_array)
                for group_index in np.unique(current_sequence_test_category_array):
                    indices = np.where(current_sequence_test_category_array == group_index)
                    tokens = list(map(dsm.id2token.get, indices[0].tolist()))
                    mean_group_activation_dict_list[1][str(group_index)].append(np.mean(vector[indices]))
                    sum_group_activation_dict_list[1][str(group_index)].append(np.sum(vector[indices]))
                    if target_df.empty != True:
                        results = target_df.loc[target_df['target2'].isin(tokens)]['sd_result'].to_list()
                        correct_number = 0
                        for result in results:
                            if result == 'hit' or result == 'cr':
                                correct_number +=1
                        guess_accuracy_dict_list[1][str(group_index)].append(correct_number/len(results))
            elif dsm.id2token[document[i]][0] == 'B':
                # print(current_sequence_test_category_array)
                for group_index in np.unique(current_sequence_test_category_array):
                    indices = np.where(current_sequence_test_category_array == group_index)
                    tokens = list(map(dsm.id2token.get, indices[0].tolist()))
                    mean_group_activation_dict_list[2][str(group_index)].append(np.mean(vector[indices]))
                    sum_group_activation_dict_list[2][str(group_index)].append(np.sum(vector[indices]))
                    if target_df.empty != True:
                        results = target_df.loc[target_df['target2'].isin(tokens)]['sd_result'].to_list()
                        correct_number = 0
                        for result in results:
                            if result == 'hit' or result == 'cr':
                                correct_number +=1
                        guess_accuracy_dict_list[2][str(group_index)].append(correct_number/len(results))
                current_sentence_index+=1


    # average the output activation, guess accuray results over all sentences
    for index,group_activation_dict in enumerate(mean_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists),axis=0)
        mean_group_activation_dict_list[index]['checkpoint']= checkpoint
        
    for index,group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists),axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists),axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list




# def test_model(self, target_context_pairs, current_epoch):
#         self.model.eval()
        
#         dist_matrix = None

#         for x in range(3):
#             if x == 0:
#                 dsm_weights = list(self.model.parameters())[0]
#                 dsm_weights = dsm_weights.cpu().detach().numpy()
#                 dist_matrix = np.corrcoef(dsm_weights)
#             elif x == 1:
#                 dist_matrix = np.corrcoef(self.get_hidden_output_activation()[0])
#             else:
#                 dist_matrix = self.get_hidden_output_activation()[1]

#             dsm_cohyponymtask = CohyponymTask(self.paradigmatic_word_category_dict, dist_matrix , self.vocab_dict)
#             best_threhold = dsm_cohyponymtask.target_balanced_accuracy_df.best_overall_threshold[0]
#             guess_accuracy_df_best_threshold = dsm_cohyponymtask.guess_accuracy_df.loc[dsm_cohyponymtask.guess_accuracy_df.threshold == best_threhold]
#             vocab_array = np.array(list(self.vocab_dict.keys()))

#             temp_mean_group_activation_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
#             temp_sum_group_activation_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
#             temp_guess_accuracy_dict_list = [defaultdict(list),defaultdict(list),defaultdict(list)]
#             current_sentence_index = 0
#             # with torch.no_grad():
#             last_center = ''
#             for i, data in enumerate(self.gen_batches(target_context_pairs), 0):
#                 center, context = data
#                 center, context = center.to(self.device), context.to(self.device)
#                 self.optimizer.zero_grad()
#                 logits, hidden = self.model(input=center)
#                 o_prob = torch.nn.functional.softmax(logits, dim=1)
#                 current_center = self.id2token[center.item()]
#                 vector = None
#                 input_index_list = []
#                 if x !=2 :
#                     vector = softmax(dist_matrix[center])
#                     input_index_list = [2, 0, 1]
#                 else:
#                     vector = o_prob.cpu().detach().numpy()[0]
#                     input_index_list = [0, 1, 2]
#                 if self.id2token[center.item()][0] == 'A':
#                     current_sequence_test_category_array = np.array(self.sequence_test_category_list[current_sentence_index])
#                     # print(current_sequence_test_category_array)
#                     for group_number in np.unique(current_sequence_test_category_array):
#                         # print(np.where(current_sequence_test_category_array == group_number))
#                         indices = np.where(current_sequence_test_category_array == group_number)
#                         temp_mean_group_activation_dict_list[input_index_list[0]][group_number].append(np.mean(vector[indices]))
#                         temp_sum_group_activation_dict_list[input_index_list[0]][group_number].append(np.sum(vector[indices]))
#                         # print(np.mean(activation_vector[0][indices]))
#                         temp_df = guess_accuracy_df_best_threshold.loc[guess_accuracy_df_best_threshold.target1 == self.id2token[center.item()]]
#                         target2_lists = list(vocab_array[indices])
#                         # print('target2_lists: {}'.format(target2_lists))
#                         temp_guess_list = temp_df.loc[temp_df['target2'].isin(target2_lists)]['guess'].to_list()
#                         print('group_number : {}  and  number of words in that category : {}'.format(group_number,len(temp_guess_list)))
#                         if group_number == 1 or group_number == 2 or group_number == 3:
#                             num_hit = 0
#                             for guess in temp_guess_list:
#                                 if guess == 1:
#                                     num_hit += 1
#                             temp_guess_accuracy_dict_list[0][group_number].append(num_hit/len(temp_guess_list))
                            
#                         else:
#                             num_cr = 0
#                             for guess in temp_guess_list:
#                                 if guess == 0:
#                                     num_cr += 1
#                             temp_guess_accuracy_dict_list[0][group_number].append(num_cr/len(temp_guess_list))
#                     if last_center == '.':
#                         # print('current A : {}, last A : {}'.format(current_A, last_A))
#                         current_sentence_index+=1
#                 elif self.id2token[center.item()][0] == 'y':
#                     current_sequence_test_category_array = np.array(self.sequence_test_category_list[current_sentence_index])
#                     # print(current_sequence_test_category_array)
#                     for group_number in np.unique(current_sequence_test_category_array):
#                         # print(np.where(current_sequence_test_category_array == group_number))
#                         indices = np.where(current_sequence_test_category_array == group_number)
#                         temp_mean_group_activation_dict_list[input_index_list[1]][group_number].append(np.mean(vector[indices]))
#                         temp_sum_group_activation_dict_list[input_index_list[1]][group_number].append(np.sum(vector[indices]))
#                         # print(np.mean(activation_vector[0][indices]))
#                         temp_df = guess_accuracy_df_best_threshold.loc[guess_accuracy_df_best_threshold.target1 == self.id2token[center.item()]]
#                         target2_lists = list(vocab_array[indices])
#                         temp_guess_list = temp_df.loc[temp_df['target2'].isin(target2_lists)]['guess'].to_list()
#                         if group_number == 8:
#                             num_hit = 0
#                             for guess in temp_guess_list:
#                                 if guess == 1:
#                                     num_hit += 1
#                             temp_guess_accuracy_dict_list[1][group_number].append(num_hit/len(temp_guess_list))
#                         else:
#                             num_cr = 0
#                             for guess in temp_guess_list:
#                                 if guess == 0:
#                                     num_cr += 1
#                             temp_guess_accuracy_dict_list[1][group_number].append(num_cr/len(temp_guess_list))
#                 elif self.id2token[center.item()][0] == 'B':
#                     current_sequence_test_category_array = np.array(self.sequence_test_category_list[current_sentence_index])
#                     # print(current_sequence_test_category_array)
#                     for group_number in np.unique(current_sequence_test_category_array):
#                         # print(np.where(current_sequence_test_category_array == group_number))
#                         indices = np.where(current_sequence_test_category_array == group_number)
#                         temp_mean_group_activation_dict_list[input_index_list[2]][group_number].append(np.mean(vector[indices]))
#                         temp_sum_group_activation_dict_list[input_index_list[2]][group_number].append(np.sum(vector[indices]))
#                         # print(np.mean(activation_vector[0][indices]))
#                         temp_df = guess_accuracy_df_best_threshold.loc[guess_accuracy_df_best_threshold.target1 == self.id2token[center.item()]]
#                         target2_lists = list(vocab_array[indices])
#                         temp_guess_list = temp_df.loc[temp_df['target2'].isin(target2_lists)]['guess'].to_list()
#                         if group_number == 5 or group_number == 6:
#                             num_hit = 0
#                             for guess in temp_guess_list:
#                                 if guess == 1:
#                                     num_hit += 1
#                             temp_guess_accuracy_dict_list[2][group_number].append(num_hit/len(temp_guess_list))
#                         else:
#                             num_cr = 0
#                             for guess in temp_guess_list:
#                                 if guess == 0:
#                                     num_cr += 1
#                             temp_guess_accuracy_dict_list[2][group_number].append(num_cr/len(temp_guess_list))
#                 last_center = current_center