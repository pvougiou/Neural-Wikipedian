# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import cPickle as pickle

dataset_file_location = '../D1/processed/with-Surface-Form-Tuples/splitDataset.p' # In case you wish to load D1.
# dataset_file_location = '../D2/processed/with-Surface-Form-Tuples/splitDataset.p' # In case you wish to load D2.

with open(dataset_file_location, 'rb') as f:
    dataset = pickle.load(f)
    
    for i in range(0, len(dataset['train']['summary_with_surf_forms_and_types'])):
        print(dataset['train']['summary_with_surf_forms_and_types'][i].replace('<start> ', '').replace(' <end>', '').encode('utf-8'))
