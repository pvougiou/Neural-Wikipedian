#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import cPickle as pickle
import numpy as np
import h5py
import random
import pandas as pd

# IMPORTANT: Make sure the parameters below match the specification of the generated
# summaries (i.e. the params['summaries_filename'] variable) in terms of the state and
# and the dataset (i.e. params['dataset_location']) that will be loaded.
params = {
    'state': 'test',
    # 'state': 'validate',
    'dataset_location': '../../D1/processed/with-Surface-Form-Tuples/',
    'summaries_filename': './checkpoints/D1/surf_form_tuples.model.t7.summaries_Testing.beam_5.h5',
}

# The location that the output .csv will be stored.
summaries_dump_location = params['summaries_filename'].replace('h5', 'csv')
# IMPORTANT: Leave the batch size unchanged
# It's the one with which we trained the models, and it should be the same
# with the one of the loaded pre-trained model that was used to generate the summaries
# (i.e. with beam-sample.lua). Change only if you train your own models using a
# different batch size.
batch_size = 85
# We are only be displaying the most probable summary.
beamidx = 0

print('Parameters')
for key in params:
    print('%s: %s' % (key, params[key]))

if 'with-Surface-Form-Tuples/' in params['dataset_location']:
    summaries_type = 'summary_with_surf_forms'
elif 'with-URIs/' in params['dataset_location']:
    summaries_type = 'summary_with_URIs'


summaries = h5py.File(params['summaries_filename'], 'r')


summaries_dictionary = params['dataset_location'] + 'summaries_dictionary.json'
with open(summaries_dictionary, 'r') as f:
    dictionary = json.load(f, 'utf-8')
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    
# Loading supporting inverse dictionaries for surface forms and instance types.
with open(params['dataset_location'] + 'inv_surf_forms_dictionary.json', 'r') as f:
    inv_surf_forms_tokens = json.load(f, encoding='utf-8')
with open(params['dataset_location'] + 'inv_instance_types_with_predicates.json', 'r') as f:
    inv_instancetypes_with_pred_dict = json.load(f, encoding='utf-8')
with open(params['dataset_location'] + 'splitDataset.p', 'rb') as f:
    splitDataset = pickle.load(f)
with open(params['dataset_location'] + 'inv_instance_types_dictionary.json', 'r') as f:
    inv_instancetypes_dict = json.load(f, encoding='utf-8')
    
if summaries_type == 'summary_with_URIs':
    with open(params['dataset_location'] + 'uri_counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)
elif summaries_type == 'summary_with_surf_forms':
    with open(params['dataset_location'] + 'surf_forms_counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)


print('All relevant dataset files from: %s have been successfully loaded.' % params['dataset_location'])

most_frequent_surf_form = {}
for entity in surf_form_counts:
    most_frequent_surf_form[entity] = sorted(surf_form_counts[entity], key=lambda k: surf_form_counts[entity][k], reverse=True)[0]



def match_predicate_to_entity(token, triples, triples_with_instance_types, expressed_triples):
    matched_entities_indices = []
    match_object_flag = True
    
    if '__obj__' in token:
        stripped = [s.strip() for s in token.split('__obj__') if s.strip()]
    if '__sub__' in token:
        stripped = [s.strip() for s in token.split('__sub__') if s.strip()]
        match_object_flag = False
    
    for tr in range(0, len(triples_with_instance_types)):
        if tr not in expressed_triples:
            tempPredicate = triples[tr].split()[1]

            if tempPredicate == stripped[0]:
                tempEntityType = triples_with_instance_types[tr].split()[-1] if match_object_flag is True else triples_with_instance_types[tr].split()[0]
                if tempEntityType not in matched_entities_indices and tempEntityType == stripped[-1]:
                    matched_entities_indices.append(tr)

    if len(matched_entities_indices) == 0:
        token = stripped[-1]
    else:
        
        random_selection = random.choice(matched_entities_indices)
        tempEntity = triples[random_selection].split()[-1].decode('utf-8') if match_object_flag is True else triples[random_selection].split()[0].decode('utf-8')
        while tempEntity not in surf_form_counts and len(matched_entities_indices) > 1:
            matched_entities_indices.remove(random_selection)
            random_selection = random.choice(matched_entities_indices)
            tempEntity = triples[random_selection].split()[-1] if match_object_flag is True else triples[random_selection].split()[0]
        
        if tempEntity in surf_form_counts:
        
            token = most_frequent_surf_form[tempEntity]
            expressed_triples.append(random_selection)
        else: 
            token = stripped[-1]
        
    return token



def token_to_word(token, main_entity, triples, triples_with_instance_types, expressed_triples):
    global summaries_type

    main_entity = main_entity.decode('utf-8')
    if "#surFormToken" in token:
        word = inv_surf_forms_tokens[token[1:]][1] if "##surFormToken" in token else inv_surf_forms_tokens[token][1]
    elif "#instanceTypeWithPredicate" in token:
        word = match_predicate_to_entity(inv_instancetypes_with_pred_dict[token], triples, triples_with_instance_types, expressed_triples)
    elif "#instanceType" in token:
        word = inv_instancetypes_dict[token]
    elif token == "<item>":
        word = most_frequent_surf_form[main_entity]
    else:
        word = token
    return word


output = {'Main-Item': [], 'Triples': [], 'Actual-Summary': [], 'Generated-Summary': []}

for batchidx in range(0, len(summaries['triples'])):
    print('Post-processing summaries from %d. Batch...' % (batchidx + 1))
    for instance in range(0, batch_size):
        # Pay attention to the Python division at the np.round() function -- can seriously mess things up!
        splitDatasetIndex = int(np.round(instance * len(splitDataset[params['state']]['item']) / float(batch_size)) + batchidx)
        mainItem = splitDataset[params['state']]['item'][splitDatasetIndex]
        output['Main-Item'].append(mainItem)

        triples = ''
        for tr in range(0, len(splitDataset[params['state']]['triples'][splitDatasetIndex])):
            triples += splitDataset[params['state']]['triples'][splitDatasetIndex][tr].replace('<item>', mainItem).decode('utf-8') + '\n'
        output['Triples'].append(triples)

        output['Actual-Summary'].append(splitDataset[params['state']]['actual_target'][splitDatasetIndex])
        
        expressed_triples = []
        summary = ''
        i = 0
        while summaries['summaries'][beamidx][batchidx * batch_size + instance][i] != word2id['<end>']:
            summary += ' ' + token_to_word(id2word[summaries['summaries'][beamidx][batchidx * batch_size + instance][i]],
                                           mainItem,
                                           splitDataset[params['state']]['triples'][splitDatasetIndex],
                                           splitDataset[params['state']]['triples_with_only_types'][splitDatasetIndex],
                                           expressed_triples)
            if i == len(summaries['summaries'][beamidx][batchidx * batch_size + instance]) - 1:
                break
            else:
                i += 1       
        summary += ' ' + token_to_word(id2word[summaries['summaries'][beamidx][batchidx * batch_size + instance][i]],
                                       mainItem,splitDataset[params['state']]['triples'][splitDatasetIndex],
                                       splitDataset[params['state']]['triples_with_only_types'][splitDatasetIndex],
                                       expressed_triples)
 
        output['Generated-Summary'].append(summary[1:])
        
# Saving all the generated summaries along with their input triples in a pandas DataFrame.
out_df = pd.DataFrame(output)
out_df.to_csv(summaries_dump_location, index=False, encoding = 'utf-8')
print('The generated summaries have been successfully saved at: %s' % summaries_dump_location)
