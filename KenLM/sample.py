#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#

import kenlm
import cPickle as pickle
import numpy as np
import copy

# Select the size of the beam for beam-sampling.
beam_size = 10


# IMPORTANT: Make sure the parameters of the variables about the location of the dataset and the trained
# model match each other's specifications in terms of the loaded dataset (i.e. D1 and D2).

trained_model_location = './D1.surf_form_tuples.model.klm'
# trained_model_location = './D2.surf_form_tuples.model.klm'

templates_dump_location = './templates/D1.surf_form_tuples.templates.p' # In case a model trained on D1 is loaded
# templates_dump_location = './templates/D2.surf_form_tuples.templates.p' # In case a model trained on D2 is loaded

dataset_file_location = '../D1/processed/with-Surface-Form-Tuples/splitDataset.p' # In case a model trained on D1 is loaded
# dataset_file_location = '../D2/processed/with-Surface-Form-Tuples/splitDataset.p' # In case a model trained on D2 is loaded

# Loading trained the trained n-gram model along with the original dataset file.
model = kenlm.LanguageModel(trained_model_location)
print('Trained KenLM model has been successfully loaded from: %s' % (trained_model_location))

with open(dataset_file_location, 'rb') as f:
    dataset = pickle.load(f)
print('Dataset file has been successfully loaded from: %s' % (dataset_file_location))

# Building vocabulary

vocab_len = 0
token2id = {}
id2token = {}
for i in range(0, len(dataset['train']['summary_with_surf_forms_and_types'])):
    tempSummary = dataset['train']['summary_with_surf_forms_and_types'][i].encode('utf-8').split()
    for token in tempSummary:
        if token not in token2id:
            vocab_len += 1
            token2id[token] = vocab_len
            id2token[vocab_len] = token
vocab_len += 1
token2id['</s>'] = vocab_len
id2token[vocab_len] = '</s>'

# Make sure that sequences are absolved from start- and end-of-sequence tokens.
# These are not handled very well in this model.
assert('<start>' not in token2id)
assert('<end>' not in token2id)


# Beam-search decoding on the trained n-gram model
# We initialise the beams with the n most probable words given the "<s>" token (i.e. "<start>" token
# in KenLM Language Model Toolkit)
# More information at: https://kheafield.com/code/kenlm/

sentences = []
sentences_prob = []
candidates = []
num_active_beams = beam_size

print('Sampling the most probable templates; depending on your system, this process might take some time...')
beam_probabilities = np.zeros(vocab_len) 
for token in token2id:
    tempCandidate = ' '.join([] + [token]) 
    beam_probabilities[token2id[token] - 1] = model.score(tempCandidate, eos = False)
indices = np.argsort(beam_probabilities)[-num_active_beams:]
for j in range(beam_size - 1, -1, -1):
    candidates.append([id2token[indices[j] + 1]])

while num_active_beams > 0:
    beam_probabilities = np.zeros(num_active_beams * vocab_len)
    beam_probabilities.fill(np.NINF)
    for s in range(0, num_active_beams):
        for token in token2id:
            tempCandidate = ' '.join(candidates[s] + [token])
            beam_probabilities[s * vocab_len + token2id[token] - 1] = model.score(tempCandidate, eos = False)
        
    indices = np.argsort(beam_probabilities)[-num_active_beams:]
    cloned_candidates = copy.deepcopy(candidates)
    completed_beams_counter = 0
    candidates = []
    for j in range(num_active_beams - 1, -1, -1):
        
        candidates.append([])
        candidates[-1] = copy.deepcopy(cloned_candidates[indices[j] / vocab_len])
        
        candidates[-1] += [id2token[indices[j] % vocab_len + 1]]
        if id2token[indices[j] % vocab_len + 1] == '</s>':
            completed_beams_counter += 1
            sentences.append(candidates[-1])
            sentences_prob.append(beam_probabilities[indices[j]])
            candidates.pop(-1)
    num_active_beams -= completed_beams_counter

print('The most probable templates along with their logarithmic probabilities are presented below:')
for s in range(0, len(sentences)):
    assert(sentences[s][-1] == '</s>')
    sentences[s] = ' '.join(sentences[s][:-1]).decode('utf-8')
    print(sentences[s])
    print('Log-probability: %.3f' % sentences_prob[s])


# Computing probabilities
# KenLM scores are log-probabilities with a base of 10; we compute the actual ones, and we normalise.

nominator = np.power(10, np.asarray(sentences_prob))
denominator = np.power(10, np.asarray(sentences_prob)).sum()
prob_distribution = nominator / denominator


# Saving everything in a dictionary
with open(templates_dump_location, 'wb') as f:
    pickle.dump({'sentences': sentences, 'prob_distribution': prob_distribution}, f)
print('The sampled summary templates have been successfully saved at: %s' % templates_dump_location)

