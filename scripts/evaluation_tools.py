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
        old_h = h.detach()
        output_activations[i] = o_prob.detach().numpy()

    # get hidden representations from srn
    hidden_dict = defaultdict(list)
    hidden_matrix = np.zeros([srn.vocab_size, srn.params.embed_size], float)
    old_h = srn.model.init_hidden_state()
    for document in srn.numeric_document_list:
        for i in range(len(document)):
            input_vector = srn.model.embed[document[i]]
            z_h, h, z_o, o_prob = srn.model.forward_item(input_vector, old_h)
            old_h = h.detach()
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
    for document in w2v.numeric_document_list:
        for i in range(len(document)):
            logits, hidden, o_prob = w2v.model(input=torch.tensor([document[i]]))
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
    test_list = [i for i in range(gpt.vocab_size)]
    test_list.append(15)
    for contexts, targets in gpt.get_batch([test_list]):
        logits, loss, o_prob = gpt.model(contexts, targets)
        if window_move == 0:
            # For the first block, get all activations
            activations.extend(o_prob.detach().tolist())
        else:
            # For subsequent blocks, get only the last activation
            activations.append(o_prob[-1].detach().tolist())
        window_move += 1
    output_activations = np.array(activations)

    # get multi-attention head outputs from gpt
    input_activations, hidden_activations, multihead_outputs, attention_weights, inputs, o_probs = \
        gpt.get_embeds(gpt.numeric_document_list)
    # Process captured data in multihead_outputs to update the dictionaries
    input_shape = input_activations[0].shape[2:]
    hidden_shape = hidden_activations[0].shape[2:]
    output_shape = multihead_outputs[0].shape[2:]
    attention_shape = attention_weights[0].shape[2:]

    accumulated_inputs = defaultdict(lambda: torch.zeros(input_shape))
    accumulated_hiddens = defaultdict(lambda: torch.zeros(hidden_shape))
    accumulated_attention_outputs = defaultdict(lambda: torch.zeros(output_shape))
    accumulated_attention_weights = defaultdict(lambda: torch.zeros(attention_shape))
    accumulated_oprobs = defaultdict(lambda: torch.zeros(output_shape))
    counts = defaultdict(int)
    for i, batch_attention_output_data in enumerate(multihead_outputs):
        batch_input_data = inputs[i]
        batch_attention_weight_data = attention_weights[i]
        batch_input_activation_data = input_activations[i]
        batch_hidden_activation_data = hidden_activations[i]
        for b in range(batch_attention_output_data.shape[0]):  # Loop over the batch dimension
            input_data = batch_input_data[b]
            attention_output_data = batch_attention_output_data[b]
            attention_weight_data = batch_attention_weight_data[b]
            input_activation_data = batch_input_activation_data[b]
            hidden_activation_data = batch_hidden_activation_data[b]
            for j, item in enumerate(input_data):  # Loop over the sequence/window
                accumulated_attention_outputs[input_data[j].item()] += attention_output_data[j]
                accumulated_attention_weights[input_data[j].item()] += attention_weight_data[j]
                accumulated_inputs[input_data[j].item()] += input_activation_data[j]
                accumulated_hiddens[input_data[j].item()] += hidden_activation_data[j]
                counts[input_data[j].item()] += 1

    # Compute the averages
    temp_averaged_outputs = {}
    temp_averaged_weights = {}
    temp_averaged_input_activation = {}
    temp_averaged_hidden_activation = {}
    for item, tensor in accumulated_attention_outputs.items():
        temp_averaged_outputs[item] = tensor / counts[item]
        temp_averaged_weights[item] = accumulated_attention_weights[item] / counts[item]
        temp_averaged_input_activation[item] = accumulated_inputs[item] / counts[item]
        temp_averaged_hidden_activation[item] = accumulated_hiddens[item] / counts[item]

    averaged_outputs = dict(sorted(temp_averaged_outputs.items()))
    averaged_weights = dict(sorted(temp_averaged_weights.items()))
    averaged_inputs = dict(sorted(temp_averaged_input_activation.items()))
    averaged_hiddens = dict(sorted(temp_averaged_hidden_activation.items()))
    averaged_outputs_matrix = np.array(list(averaged_outputs.values()))
    averaged_weights_matrix = np.array(list(averaged_weights.values()))
    averaged_inputs_matrix = np.array(list(averaged_inputs.values()))
    averaged_hiddens_matrix = np.array(list(averaged_hiddens.values()))
    return output_activations, averaged_outputs_matrix, averaged_weights_matrix, \
           averaged_inputs_matrix, averaged_hiddens_matrix


def evaluate_srn(checkpoint, srn: SRN, cohyponymtask: CohyponymTask):
    mean_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sim_score_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    srn.model.eval()
    # get output activations from srn
    old_h = srn.model.init_hidden_state()
    guess_accuracy_best_df = cohyponymtask.guess_accuracy_best_df
    for document in srn.numeric_document_list:
        sentence_count = 0
        for i in range(len(document[:-1])):
            token = srn.decode([document[i]])[0]
            df = guess_accuracy_best_df.loc[cohyponymtask.guess_accuracy_best_df.target1 == token]
            input_vector = srn.model.embed[document[i]]
            z_h, h, z_o, o_prob = srn.model.forward_item(input_vector, old_h)
            old_h = h.detach()
            output_activations = o_prob.cpu().detach().numpy()[0]
            save_index = 0 if token[0] == 'A' else 1 if token[0] == 'y' else 2 if token[0] == 'B' else 3
            current_sequence_test_category_array = np.array(srn.corpus.sequence_test_category_list[sentence_count])
            for group_index in np.unique(current_sequence_test_category_array):
                # print(np.where(current_sequence_test_category_array == group_index))
                # get the index of tokens in the word_dict
                indices = np.where(current_sequence_test_category_array == group_index)
                tokens = list(map(srn.id_vocab_dict.get, indices[0].tolist()))
                results = df.loc[df['target2'].isin(tokens)]['sd_result'].to_list()
                sim_scores = np.array(df.loc[df['target2'].isin(tokens)]['score'].to_list())
                correct_number = 0
                for result in results:
                    if result == 'hit' or result == 'cr':
                        correct_number += 1
                mean_group_activation_dict_list[save_index][str(group_index)].append(np.mean(output_activations[indices]))
                sum_group_activation_dict_list[save_index][str(group_index)].append(np.sum(output_activations[indices]))
                guess_accuracy_dict_list[save_index][str(group_index)].append(correct_number / len(results))
                sim_score_dict_list[save_index][str(group_index)].append(np.mean(sim_scores))
            sentence_count = sentence_count + 1 if token[0] == '.' else sentence_count

    # average the output activation, guess accuray results over all sentences
    for index, group_activation_dict in enumerate(mean_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        mean_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists), axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint
    for index, sim_score_dict in enumerate(sim_score_dict_list):
        for group_index, sim_score_lists in sim_score_dict.items():
            sim_score_dict_list[index][str(group_index)] = np.mean(np.array(sim_score_lists), axis=0)
        sim_score_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list, sim_score_dict_list


def evaluate_lstm(checkpoint, lstm: LSTM, cohyponymtask: CohyponymTask):
    mean_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sim_score_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    lstm.model.eval()
    # get output activations from LSTM
    old_h, old_c = lstm.model.init_hidden_state_2d()
    guess_accuracy_best_df = cohyponymtask.guess_accuracy_best_df
    for document in lstm.numeric_document_list:
        sentence_count = 0
        for i in range(len(document[:-1])):
            token = lstm.decode([document[i]])[0]
            df = guess_accuracy_best_df.loc[cohyponymtask.guess_accuracy_best_df.target1 == token]
            z_o, o_prob, h, c = lstm.model(document[i], old_h, old_c)
            old_h, old_c = h.detach(), c.detach()
            output_activations = o_prob.cpu().detach().numpy()[0]
            save_index = 0 if token[0] == 'A' else 1 if token[0] == 'y' else 2 if token[0] == 'B' else 3
            current_sequence_test_category_array = np.array(lstm.corpus.sequence_test_category_list[sentence_count])
            for group_index in np.unique(current_sequence_test_category_array):
                # print(np.where(current_sequence_test_category_array == group_index))
                # get the index of tokens in the word_dict
                indices = np.where(current_sequence_test_category_array == group_index)
                tokens = list(map(lstm.id_vocab_dict.get, indices[0].tolist()))
                results = df.loc[df['target2'].isin(tokens)]['sd_result'].to_list()
                sim_scores = np.array(df.loc[df['target2'].isin(tokens)]['score'].to_list())
                correct_number = 0
                for result in results:
                    if result == 'hit' or result == 'cr':
                        correct_number += 1
                mean_group_activation_dict_list[save_index][str(group_index)].append(np.mean(output_activations[indices]))
                sum_group_activation_dict_list[save_index][str(group_index)].append(np.sum(output_activations[indices]))
                guess_accuracy_dict_list[save_index][str(group_index)].append(correct_number / len(results))
                sim_score_dict_list[save_index][str(group_index)].append(np.mean(sim_scores))
            sentence_count = sentence_count + 1 if token[0] == '.' else sentence_count

    # average the output activation, guess accuray results over all sentences
    for index, group_activation_dict in enumerate(mean_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        mean_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists), axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint
    for index, sim_score_dict in enumerate(sim_score_dict_list):
        for group_index, sim_score_lists in sim_score_dict.items():
            sim_score_dict_list[index][str(group_index)] = np.mean(np.array(sim_score_lists), axis=0)
        sim_score_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list, sim_score_dict_list


def evaluate_w2v(checkpoint, w2v: W2Vec, cohyponymtask: CohyponymTask):
    mean_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sim_score_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]

    w2v.model.eval()
    guess_accuracy_best_df = cohyponymtask.guess_accuracy_best_df
    for document in w2v.numeric_document_list:
        sentence_count = 0
        for i in range(len(document[:-1])):
            token = w2v.decode([document[i]])[0]
            df = guess_accuracy_best_df.loc[cohyponymtask.guess_accuracy_best_df.target1 == token]
            logits, hidden, o_prob = w2v.model(input=torch.tensor([document[i]]))
            output_activations = o_prob.cpu().detach().numpy()[0]
            save_index = 0 if token[0] == 'A' else 1 if token[0] == 'y' else 2 if token[0] == 'B' else 3
            current_sequence_test_category_array = np.array(w2v.corpus.sequence_test_category_list[sentence_count])
            for group_index in np.unique(current_sequence_test_category_array):
                # print(np.where(current_sequence_test_category_array == group_index))
                # get the index of tokens in the word_dict
                indices = np.where(current_sequence_test_category_array == group_index)
                tokens = list(map(w2v.id_vocab_dict.get, indices[0].tolist()))
                results = df.loc[df['target2'].isin(tokens)]['sd_result'].to_list()
                sim_scores = np.array(df.loc[df['target2'].isin(tokens)]['score'].to_list())
                correct_number = 0
                for result in results:
                    if result == 'hit' or result == 'cr':
                        correct_number += 1
                mean_group_activation_dict_list[save_index][str(group_index)].append(np.mean(output_activations[indices]))
                sum_group_activation_dict_list[save_index][str(group_index)].append(np.sum(output_activations[indices]))
                guess_accuracy_dict_list[save_index][str(group_index)].append(correct_number / len(results))
                sim_score_dict_list[save_index][str(group_index)].append(np.mean(sim_scores))
            sentence_count = sentence_count + 1 if token[0] == '.' else sentence_count

    # average the output activation, guess accuray results over all sentences
    for index, group_activation_dict in enumerate(mean_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        mean_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists), axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint
    for index, sim_score_dict in enumerate(sim_score_dict_list):
        for group_index, sim_score_lists in sim_score_dict.items():
            sim_score_dict_list[index][str(group_index)] = np.mean(np.array(sim_score_lists), axis=0)
        sim_score_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list, sim_score_dict_list



def evaluate_gpt(checkpoint, gpt: GPT, cohyponymtask: CohyponymTask):
    mean_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    sim_score_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    gpt.model.eval()

    is_first = True
    sentence_count = 0
    guess_accuracy_best_df = cohyponymtask.guess_accuracy_best_df
    for contexts, targets in gpt.get_batch(gpt.numeric_document_list):
        logits, loss, o_prob = gpt.model(contexts, targets)
        for batch in contexts:
            if is_first:
                words = gpt.decode(batch.tolist())
                word_idices = [i for i in range(len(words))]
                is_first = False
            else:
                words = gpt.decode([batch[-1].item()])
                word_idices = [batch.size(0)-1]
            for (token, index) in zip(words, word_idices):
                current_sequence_test_category_array = np.array(gpt.corpus.sequence_test_category_list[sentence_count])
                df = guess_accuracy_best_df.loc[cohyponymtask.guess_accuracy_best_df.target1 == token]
                save_index = 0 if token[0] == 'A' else 1 if token[0] == 'y' else 2 if token[0] == 'B' else 3
                output_activations = o_prob[index].detach().numpy()
                for group_index in np.unique(current_sequence_test_category_array):
                    # print(np.where(current_sequence_test_category_array == group_index))
                    # get the index of tokens in the word_dict
                    indices = np.where(current_sequence_test_category_array == group_index)
                    tokens = list(map(gpt.id_vocab_dict.get, indices[0].tolist()))
                    results = df.loc[df['target2'].isin(tokens)]['sd_result'].to_list()
                    sim_scores = np.array(df.loc[df['target2'].isin(tokens)]['score'].to_list())
                    correct_number = 0
                    for result in results:
                        if result == 'hit' or result == 'cr':
                            correct_number += 1
                    mean_group_activation_dict_list[save_index][str(group_index)].append(
                        np.mean(output_activations[indices]))
                    sum_group_activation_dict_list[save_index][str(group_index)].append(
                        np.sum(output_activations[indices]))
                    guess_accuracy_dict_list[save_index][str(group_index)].append(correct_number / len(results))
                    sim_score_dict_list[save_index][str(group_index)].append(np.mean(sim_scores))
                    # print(guess_accuracy_dict_list[2]['5'])
                sentence_count = sentence_count + 1 if token[0] == '.' else sentence_count

    # average the output activation, guess accuray results over all sentences
    for index, group_activation_dict in enumerate(mean_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            mean_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        mean_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, group_activation_dict in enumerate(sum_group_activation_dict_list):
        for group_index, activation_lists in group_activation_dict.items():
            sum_group_activation_dict_list[index][str(group_index)] = np.mean(np.array(activation_lists), axis=0)
        sum_group_activation_dict_list[index]['checkpoint'] = checkpoint

    for index, guess_accuracy_dict in enumerate(guess_accuracy_dict_list):
        for group_index, guess_accuracy_lists in guess_accuracy_dict.items():
            guess_accuracy_dict_list[index][str(group_index)] = np.mean(np.array(guess_accuracy_lists), axis=0)
        guess_accuracy_dict_list[index]['checkpoint'] = checkpoint

    for index, sim_score_dict in enumerate(sim_score_dict_list):
        for group_index, sim_score_lists in sim_score_dict.items():
            sim_score_dict_list[index][str(group_index)] = np.mean(np.array(sim_score_lists), axis=0)
        sim_score_dict_list[index]['checkpoint'] = checkpoint
    return mean_group_activation_dict_list, sum_group_activation_dict_list, guess_accuracy_dict_list, sim_score_dict_list

def old_evaluate_model(dsm, dsm_type, checkpoint, seq_num, sequence_test_category_list, similarity_matrix=np.array([]) ,cohyponymtask: CohyponymTask =None):
    dsm.model.eval()
    if dsm_type == 'srn':
        old_h = dsm.model.init_hidden_state()
    elif dsm_type == 'lstm':
        old_h, old_c = dsm.model.init_hidden_state_2d()
    mean_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list)]
    sum_group_activation_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list)]
    guess_accuracy_dict_list = [defaultdict(list), defaultdict(list), defaultdict(list)]
    # iterate through every document
    for doc_index, document in enumerate(seq_num):
        # set the first sentence index to 0, so that we can keep track of the number of sentences we tested
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


