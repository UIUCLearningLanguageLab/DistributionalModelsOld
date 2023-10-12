import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict, OrderedDict
from distributionalmodels.tasks.cohyponym_task import CohyponymTask
from distributionalmodels.models.srn import SRN
from distributionalmodels.models.lstm import LSTM
from distributionalmodels.models.word2vec import W2Vec
from distributionalmodels.models.gpt import GPT
import pandas as pd
import torch

# get all meaningful vectors from srn
def get_representations_from_srn(srn: SRN):
    srn.model.eval()

    # get output activations from srn
    old_h = srn.model.init_hidden_state()
    output_activations = np.zeros((srn.vocab_size, srn.vocab_size))
    for i in range(srn.vocab_size):
        input_vector = srn.model.embed[i]
        z_h, h, z_o, o_prob = srn.model.forward_item(input_vector, old_h)
        old_h = h.detach
        output_activations[i] = o_prob.detach().numpy()

    # get hidden representations from srn
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([srn.vocab_size, srn.params.embed_size], float)
    old_h = srn.model.init_hidden_state()
    for document in srn.numeric_document_list:
        for i in range(len(document)):
            input_vector = srn.model.embed[document[i]]
            z_h, h, z_o, o, o_prob = srn.model.forward_item(input_vector, old_h)
            old_h = h.detach
            # store every recurrent activtion of each unique token in a dictionary
            hidden_dict[document[i]].append(old_h.numpy())

    sorted_hidden_dict = OrderedDict(sorted(hidden_dict.items()))
    # average the activations
    for vocab_index, hidden_list in sorted_hidden_dict.items():
        hidden_matrix[vocab_index] = np.mean(np.array(hidden_list), axis=0)
    return output_activations, hidden_matrix

# get all meaningful vectors from lstm
def get_representations_from_lstm(lstm: LSTM):
    lstm.model.eval()

    # get output activations from lstm
    old_h, old_c = lstm.model.init_hidden_state_2d()
    output_activations = np.zeros((lstm.vocab_size, lstm.vocab_size))

    for i in range(lstm.vocab_size):
        z_o, o_prob, h, c = lstm.model(i, old_h, old_c)
        old_h, old_c = h.detach(), c.detach()
        output_activations[i] = o_prob.detach().numpy()

    # get hidden representations from lstm
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([lstm.vocab_size, lstm.params.embed_size], float)
    old_h, old_c = lstm.model.init_hidden_state_2d()
    for document in lstm.numeric_document_list:
        for i in range(len(document)):
            z_o, o_prob, h, c = lstm.model(document[i], old_h, old_c)
            old_h, old_c = h.detach(), c.detach()
            hidden_dict[document[i]].append(old_h.numpy())

    sorted_hidden_dict = OrderedDict(sorted(hidden_dict.items()))
    # average the activations
    for vocab_index, hidden_list in sorted_hidden_dict.items():
        hidden_matrix[vocab_index] = np.mean(np.array(hidden_list), axis=0)
    return output_activations, hidden_matrix

# get all meaningful vectors from w2v
def get_representations_from_w2v(w2v: W2Vec):
    w2v.model.eval()

    # get output activation from w2v
    output_activations = np.zeros((w2v.vocab_size, w2v.vocab_size))
    for i in range(w2v.vocab_size):
        logits, hidden, o_prob = w2v.model(input=torch.tensor([i]))
        output_activations[i] = o_prob.detach().numpy()

    # get hidden activation from w2v
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([w2v.vocab_size, w2v.params.embed_size], float)
    for document in w2v.numeric_sequence_list:
        for i in range(len(document)):
            logits, hidden = w2v.model(input=torch.tensor([document[i]]))
            hidden_dict[document[i]].append(hidden.detach().numpy())

    sorted_hidden_dict = OrderedDict(sorted(hidden_dict.items()))
    # average the activations
    for vocab_index, hidden_list in sorted_hidden_dict.items():
        hidden_matrix[vocab_index] = np.mean(np.array(hidden_list), axis=0)
    return output_activations, hidden_matrix

# get all meaningful vectors from gpt
def get_representations_from_gpt(gpt: GPT):
    gpt.model.eval()

    # get output activations from gpt
    output_activations = np.zeros((gpt.vocab_size, gpt.vocab_size))
    window_move = 0
    activations = []
    for contexts, targets in gpt.get_batch([[i for i in range(gpt.vocab_size)]]):
        logits, loss, o_prob = gpt.model(contexts, targets)
        if window_move == 0:
            # For the first block, get all activations
            activations.extend(o_prob.detach().tolist())
        else:
            # For subsequent blocks, get only the last activation
            activations.append(o_prob[-1].detach().tolist())
    output_activations = np.array(activations)

    # get multi-attention head outputs from gpt
    multihead_outputs, multihead_inputs = gpt.get_multihead_outputs(gpt.numeric_document_list)
    # Process captured data in multihead_outputs to update the dictionaries
    output_shape = multihead_outputs[0].shape[2:]
    accumulated_outputs = defaultdict(lambda: torch.zeros(output_shape))
    counts = defaultdict(int)
    for i, batch_output_data in enumerate(multihead_outputs):
        batch_input_data = multihead_inputs[i]
        for b in range(batch_output_data.shape[0]):  # Loop over the batch dimension
            input_data = batch_input_data[b]
            output_data = batch_output_data[b]
            for j, item in enumerate(input_data):  # Loop over the sequence/window
                accumulated_outputs[input_data[j].item()] += output_data[j]
                counts[input_data[j].item()] += 1

    # Compute the averages
    temp_averaged_outputs = {}
    for item, tensor in accumulated_outputs.items():
        temp_averaged_outputs[item] = tensor / counts[item]
    averaged_outputs = dict(sorted(temp_averaged_outputs.items()))
    averaged_outputs_matrix = np.array(list(averaged_outputs.values()))
    print(averaged_outputs_matrix.shape)
    return output_activations, averaged_outputs_matrix







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
    if dsm_type == 'gpt':
        window_move = 0
        activations = []
        for contexts, targets in dsm.get_batch([[i for i in range(dsm.vocab_size)]]):
            logits, loss, o_prob = dsm.model(contexts, targets)
            if window_move == 0:
                # For the first block, get all activations
                activations.extend(o_prob.detach().tolist())
            else:
                # For subsequent blocks, get only the last activation
                activations.append(o_prob[-1].detach().tolist())
        output_activations = np.array(activations)
    else:
        for i in range(dsm.vocab_size):
            if dsm_type == 'srn':
                input_vector = dsm.model.embed[i]
                z_h, h, z_o, o_prob = dsm.model.forward_item(input_vector, old_h)
                old_h = h.detach()
            elif dsm_type == 'lstm':
                z_o, o_prob, h, c = dsm.model(i, old_h, old_c)
                old_h, old_c = h.detach(), c.detach()
            elif dsm_type == 'w2v':
                logits, hidden, o_prob = dsm.model(input=torch.tensor([i]))

            output_activations[i] = o_prob.detach().numpy()
    return output_activations


# stack recurrent state activation for each unique token
def get_hidden_activation(dsm, dsm_type):
    dsm.model.eval()
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([dsm.vocab_size, dsm.params.embed_size], float)
    if dsm_type == 'gpt':
        multihead_outputs, multihead_inputs = dsm.get_multihead_outputs(dsm.numeric_document_list)
        # Process captured data in multihead_outputs to update the dictionaries
        output_shape = multihead_outputs[0].shape[2:]
        accumulated_outputs = defaultdict(lambda: torch.zeros(output_shape))
        counts = defaultdict(int)
        for i, batch_output_data in enumerate(multihead_outputs):
            batch_input_data = multihead_inputs[i]
            for b in range(batch_output_data.shape[0]):  # Loop over the batch dimension
                input_data = batch_input_data[b]
                output_data = batch_output_data[b]
                for j, item in enumerate(input_data):  # Loop over the sequence/window
                    accumulated_outputs[input_data[j].item()] += output_data[j]
                    counts[input_data[j].item()] += 1

        # Compute the averages
        temp_averaged_outputs = {}
        for item, tensor in accumulated_outputs.items():
            temp_averaged_outputs[item] = tensor / counts[item]
        averaged_outputs = dict(sorted(temp_averaged_outputs.items()))
    else:
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
                input = torch.tensor([document[i]])
                logits, hidden = dsm.model(input=input)
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
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        mean_group_activation_dict_list[index]['checkpoint'] = checkpoint
        
    for index,group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists), axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list


