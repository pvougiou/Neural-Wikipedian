#!/bin/bash

# Make sure that the name of the output file (e.g. D1.surf_form_tuples.model.arpa) matches the
# loaded dataset file in the dataset.py file (i.e. dataset_location) variable.
# Comment-in and out the lines below accordingly.

python dataset.py | ./kenlm/build/bin/lmplz -o 5 > D1.surf_form_tuples.model.arpa # In case D1 is loaded in dataset.py
# python dataset.py | ./kenlm/build/bin/lmplz -o 5 > D2.surf_form_tuples.model.arpa # In case D2 is loaded in dataset.py

./kenlm/build/bin/build_binary D1.surf_form_tuples.model.arpa D1.surf_form_tuples.model.klm # In case D1 is loaded in dataset.py
# ./kenlm/build/bin/build_binary D2.surf_form_tuples.model.arpa D2.surf_form_tuples.model.klm # In case D2 is loaded in dataset.py

echo Training has been completed.

