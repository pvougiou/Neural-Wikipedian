#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import cPickle as pickle
import numpy as np
import random
import pandas as pd


# IMPORTANT: Make sure the parameters below match the specification of the sampled from the KenLM
# templates (i.e. the summary_templates_location variable) in terms of the state variable and
# and the dataset that will be loaded. Comment-in and out accordingly.

state = 'test'
# state = 'validate'
dataset_location = '../D1/processed/with-Surface-Form-Tuples/'
# dataset_location = '../D2/processed/with-Surface-Form-Tuples/'


# Loading the sampled KenLM tempates.
summary_templates_location = './templates/D1.surf_form_tuples.templates.p' if 'D1' in dataset_location else './templates/D2.surf_form_tuples.templates.p'
summaries_dump_location = summary_templates_location.replace('templates.p', 'summaries.csv')
with open(summary_templates_location, 'rb') as f:
    summary_templates = pickle.load(f)
    templates = summary_templates['sentences']
    prob_distribution = summary_templates['prob_distribution']


summaries_dictionary = dataset_location + 'summaries_dictionary.json'
with open(summaries_dictionary, 'r') as f:
    dictionary = json.load(f, 'utf-8')
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    
# Loading supporting inverse dictionaries for surface forms and instance types.
with open(dataset_location + 'inv_surf_forms_dictionary.json', 'r') as f:
    inv_surf_forms_tokens = json.load(f, encoding='utf-8')
with open(dataset_location + 'inv_instance_types_with_predicates.json', 'r') as f:
    inv_instancetypes_with_pred_dict = json.load(f, encoding='utf-8')
with open(dataset_location + 'splitDataset.p', 'rb') as f:
    splitDataset = pickle.load(f)
with open(dataset_location + 'inv_instance_types_dictionary.json', 'r') as f:
    inv_instancetypes_dict = json.load(f, encoding='utf-8')

summaries_type = 'summary_with_surf_forms'
if summaries_type == 'summary_with_URIs':
    with open(dataset_location + 'uri_counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)
elif summaries_type == 'summary_with_surf_forms':
    with open(dataset_location + 'surf_forms_counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)


print('All relevant dataset files from: %s have been successfully loaded.' % dataset_location)

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

# IMPORTANT: Leave the batch size unchanged
# It ensures that the validation will be conducted using the same input sets of triples that are used for the validation
# and testing of the neural net models.
batch_size = 85


output = {'Main-Item': [], 'Triples': [], 'Actual-Summary': [], 'Generated-Summary': []}
num_batches = int(np.floor(len(splitDataset[state]['triples']) / batch_size))
for batchidx in range(0, num_batches):
    # print('Generating summaries for %d. Batch...' % (batchidx + 1))
    for instance in range(0, batch_size):
        # Pay attention to the Python division at the np.round() function -- can seriously mess things up!
        splitDatasetIndex = int(np.round(instance * len(splitDataset[state]['item']) / float(batch_size)) + batchidx)
        mainItem = splitDataset[state]['item'][splitDatasetIndex]
        output['Main-Item'].append(mainItem)


        triples = ''
        for tr in range(0, len(splitDataset['test']['triples'][splitDatasetIndex])):
            triples += splitDataset['test']['triples'][splitDatasetIndex][tr].replace('<item>', mainItem).decode('utf-8') + '\n'
        output['Triples'].append(triples)

        output['Actual-Summary'].append(splitDataset[state]['actual_target'][splitDatasetIndex])

        assert(len(prob_distribution) == len(templates))
        rand_instance = np.random.choice(np.arange(len(prob_distribution)), p=np.ndarray.tolist(prob_distribution))
        # Comment-out the line below and in the one above should you want to generate summaries only from the most
        # probable template.
        # rand_instance = np.argmax(prob_distribution)            

        expressed_triples = []
        summary = ['<start>']
        tempSummary = templates[rand_instance]
        for word in tempSummary.split():
            summary.append(token_to_word(word,
                                         mainItem,
                                         splitDataset[state]['triples'][splitDatasetIndex],
                                         splitDataset[state]['triples_with_only_types'][splitDatasetIndex],
                                         expressed_triples))
        summary = summary + ['<end>']
        output['Generated-Summary'].append(' '.join(summary))
            
# Saving all the generated summaries along with their input triples in a pandas DataFrame.
out_df = pd.DataFrame(output)
out_df.to_csv(summaries_dump_location, index=False, encoding = 'utf-8')
print('The generated summaries have been successfully saved at: %s' % summaries_dump_location)
