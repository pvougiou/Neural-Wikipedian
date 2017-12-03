# Neural-Wikipedian
This repository contains the code along with the datasets of the work that has been submitted as a research paper to the Journal of Web Semantics. The work focuses on how an adaptation of the encoder-decoder framework can be used to generate textual summaries for Semantic Web triples.

For a detailed description of the work presented in this repository, please refer to the preprint version of the submitted paper at: <https://arxiv.org/abs/1711.00155>.

## Datasets
In order to train our proposed models, we built two datasets of aligned knowledge base triples with text. 

 * D1: DBpedia triples aligned with Wikipedia biographies
 * D2: Wikidata triples aligned with Wikipedia biographies

In a Unix shell environment execute: `sh download_datasets.sh` in order to download and uncompress both of them in their corresponding folders (i.e. `D1` and `D2`). Each dataset folder consists of three different sub-folders:

* `data` contains each aligned dataset in binary-encoded `pickle` files. Each file is a hash table. Each hash table is a Python dictionary of lists.
* `utils` contains each dataset's supporting files, such as hash tables of the frequency with which surface forms in the Wikipedia summaries have been mapped to entity URIs. All the files are binary-encoded in `pickle` files.
* `processed` contains the processed version of each aligned dataset after removal of potential outliers (e.g. instances of the datasets with extremely long Wikipedia summaries or very few triples). The files that are contained in the `processed` folders are the ones that are used for the training and testing of both our neural-network-based systems and the baselines.

[`Inspect-Dataset.ipynb`](Inspect-Dataset.ipynb) is a Python script on iPython Notebook that allows easier inspection of the above aligned datasets. The scripts provides also detailed information regarding the structure of the intermediate parts in `D1/data/` and `D2/data/` and the functionality of the supporting files in `D1/utils/` and `D2/utils/`.

The table below presents the distribution of the 10 most common predicates, and entities in our two datasets, D1 and D2 respectively.

<table>
  <thead>
    <tr>
      <td><b>Predicates In Triples</b></td>
      <td align="center">%</td>
      <td><b>Entities In Triples</b></td>
      <td align="center">%</td>
      <td><b>Entities In Summaries</b></td>
      <td align="center">%</td>
    </tr>
  </thead>
  <tr>
    <td><tt>dbo:birthDate</tt></td>
    <td align="center">12.43</td>
    <td><tt>dbr:United_States</tt></td>
    <td align="center">0.49</td>
    <td><tt>dbr:United_States</tt></td>
    <td align="center">2.82</td>
  </tr>
  <tr>
    <td><tt>dbo:birthPlace</tt></td>
    <td align="center">10.67</td>
    <td><tt>dbr:England</tt></td>
    <td align="center">0.19</td>
    <td><tt>dbr:Actor</tt></td>
    <td align="center">2.14</td>
  </tr>
  <tr>
    <td><tt>dbo:careerStation</tt></td>
    <td align="center">5.47</td>
    <td><tt>dbr:United_Kingdom</tt></td>
    <td align="center">0.14</td>
    <td><tt>dbr:Association_football</tt></td>
    <td align="center">1.02</td>
  </tr>
  <tr>
    <td><tt>dbo:deathDate</tt></td>
    <td align="center">5.11</td>
    <td><tt>dbr:France</tt></td>
    <td align="center">0.14</td>
    <td><tt>dbr:Politician</tt></td>
    <td align="center">0.97</td>
  </tr>
  <tr>
    <td><tt>dbo:occupation</tt></td>
    <td align="center">5.06</td>
    <td><tt>dbr:Canada</tt></td>
    <td align="center">0.12</td>
    <td><tt>dbr:Singing</tt></td>
    <td align="center">0.90</td>
  </tr>
  <tr>
    <td><tt>dbo:team</tt></td>
    <td align="center">4.18</td>
    <td><tt>dbr:India</tt></td>
    <td align="center">0.11</td>
    <td><tt>dbr:United_Kingdom</tt></td>
    <td align="center">0.59</td>
  </tr>
  <tr>
    <td><tt>dbo:deathPlace</tt></td>
    <td align="center">3.51</td>
    <td><tt>dbr:Actor</tt></td>
    <td align="center">0.10</td>
    <td><tt>dbr:England</tt></td>
    <td align="center">0.58</td>
  </tr>
  <tr>
    <td><tt>dbo:genre</tt></td>
    <td align="center">3.22</td>
    <td><tt>dbr:Italy</tt></td>
    <td align="center">0.10</td>
    <td><tt>dbr:Writer</tt></td>
    <td align="center">0.53</td>
  </tr>
    <tr>
    <td><tt>dbo:associatedBand</tt></td>
    <td align="center">2.85</td>
    <td><tt>dbr:London</tt></td>
    <td align="center">0.10</td>
    <td><tt>dbr:Canada</tt></td>
    <td align="center">0.50</td>
  </tr>
    <tr>
    <td><tt>dbp:associatedMusicalArtist</tt></td>
    <td align="center">2.85</td>
    <td><tt>dbr:Japan</tt></td>
    <td align="center">0.09</td>
    <td><tt>dbr:France</tt></td>
    <td align="center">0.49</td>
  </tr>
  <tr>
  <td colspan="6"></td>
  </tr>
  <thead>
    <tr>
      <td><b>Predicates In Triples</b></td>
      <td align="center">%</td>
      <td><b>Entities In Triples</b></td>
      <td align="center">%</td>
      <td><b>Entities In Summaries</b></td>
      <td align="center">%</td>
    </tr>
  </thead>
    <td><tt>wikidata:P569</tt><br/> (place of birth)</td>
    <td align="center">14.15</td>
    <td><tt>wikidata:Q5</tt><br/> (human)</td>
    <td align="center">3.96</td>
    <td><tt>wikidata:Q30</tt><br/> (United States of America)</td>
    <td align="center">3.20</td>
  </tr>
  <tr>
    <td><tt>wikidata:P106</tt><br/> (occupation)</td>
    <td align="center">11.63</td>
    <td><tt>wikidata:Q6581097</tt><br/> (male)</td>
    <td align="center">3.27</td>
    <td><tt>wikidata:Q33999</tt><br/> (actor)</td>
    <td align="center">1.56</td>
  </tr>
  <tr>
    <td><tt>wikidata:P31</tt><br/> (instance of)</td>
    <td align="center">8.29</td>
    <td><tt>wikidata:Q30</tt><br/> (United States of America)</td>
    <td align="center">1.13</td>
    <td><tt>wikidata:Q82955</tt><br/> (politician)</td>
    <td align="center">1.02</td>
  </tr>
  <tr>
    <td><tt>wikidata:P21</tt><br/> (sex or gender)</td>
    <td align="center">7.92</td>
    <td><tt>wikidata:Q6581072</tt><br/> (female)</td>
    <td align="center">0.70</td>
    <td><tt>wikidata:Q21</tt><br/> (England)</td>
    <td align="center">0.87</td>
  </tr>
  <tr>
    <td><tt>wikidata:P570</tt><br/> (date of death)</td>
    <td align="center">7.58</td>
    <td><tt>wikidata:Q145</tt><br/> (United Kingdom)</td>
    <td align="center">0.44</td>
    <td><tt>wikidata:Q145</tt><br/> (United Kingdom)</td>
    <td align="center">0.85</td>
  </tr>
  <tr>
    <td><tt>wikidata:P27</tt><br/> (country of citizenship)</td>
    <td align="center">6.75</td>
    <td><tt>wikidata:Q82955</tt><br/> (politician)</td>
    <td align="center">0.42</td>
    <td><tt>wikidata:Q27939</tt><br/> (singing)</td>
    <td align="center">0.79</td>
  </tr>
  <tr>
    <td><tt>wikidata:P735</tt><br/> (given name)</td>
    <td align="center">6.53</td>
    <td><tt>wikidata:Q1860</tt><br/> (English)</td>
    <td align="center">0.39</td>
    <td><tt>wikidata:Q36180</tt><br/> (writer)</td>
    <td align="center">0.71</td>
  </tr>
  <tr>
    <td><tt>wikidata:P19</tt><br/> (place of birth)</td>
    <td align="center">5.20</td>
    <td><tt>wikidata:Q33999</tt><br/> (actor)</td>
    <td align="center">0.36</td>
    <td><tt>wikidata:Q2736</tt><br/> (association football)</td>
    <td align="center">0.68</td>
  </tr>
    <tr>
    <td><tt>wikidata:P5</tt><br/> (member of sports team)</td>
    <td align="center">2.64</td>
    <td><tt>wikidata:Q36180</tt><br/> (writer)</td>
    <td align="center">0.24</td>
    <td><tt>wikidata:Q183</tt><br/> (Germany)</td>
    <td align="center">0.61</td>
  </tr>
    <tr>
    <td><tt>wikidata:P69</tt><br/> (educated at)</td>
    <td align="center">2.58</td>
    <td><tt>wikidata:Q177220</tt><br/> (singer)</td>
    <td align="center">0.20</td>
    <td><tt>wikidata:Q16</tt><br/> (Canada)</td>
    <td align="center">0.58</td>
  </tr>
</table>

## Our Systems
The `Systems` directory contains all the code to both train and generate summaries for the sets of triples that are located in the validation and test sets of our datasets. It contains our two models in two separate sub-folders (i.e. `Triples2GRU` and `Triples2LSTM`). The neural network models are implemented using the [Torch](http://torch.ch/) package. We conducted our experiments on a single Titan X (Pascal) GPU. Please make sure that Torch along with the [torch-hdf5](https://github.com/deepmind/torch-hdf5) package and the NVIDIA CUDA drivers are installed in your machine before executing any of the `.lua` files in these directories.

* You can train your own Triples2LSTM or Triples2GRU models, by executing `th train.lua` inside each system's directory. You need to have access to a GPU with at least 11 GB of memory in order to train the models with the same hyperparameters that we used in the paper. However, by lowering the `params.batch_size` and `params.rnn_size` variables you can train on NVIDIA GPUs will less amount of dedicated memory. By altering the `dataset_path` and `checkpoint_path` variables in each `train.lua` file, you can select the dataset (i.e. D1 or D2) on which you will be training your model, and whether you will using the surface form tuples or URIs setup. The checkpoint files of the trained models will be saved in the corresponding `checkpoints` directory.

* You can use a checkpoint of a trained model to start generating summaries given input sets of triples from the validation and test sets of the aligned datasets by executing `th beam-sample.lua`. Please make sure that the pre-trained model (i.e. on D1 or D2, with URIs or surface form tuples) matches the dataset that will be loaded in the `beam_sampling_params.dataset_path` variable. You can download all our trained models and generate summaries from them by running the shell scripts located at:
 * `Systems/Triples2LSTM/download_trained_models.sh`
 * `Systems/Triples2GRU/download_trained_models.sh`

 The generated summaries will be saved as HDF5 files in the directory of the pre-trained model. Our trained models use CUDA Tensors. Consequently, the NVIDIA CUDA drivers along with the `cutorch` and `cunn` Lua packages should be installed in your machine. The latter can be installed by running:
  ```sh
  luarocks install cutorch
  luarocks install cunn
  ```

* Execute the Python script `beam-sample.py` in order to create a `.csv` file with the sampled summaries. The following Python packages: (i) `h5py`, (ii) `pandas`, and (iii) `numpy` should be installed in your machine. The script replaces the `<item>` tokens along with the property-type placeholders, and presents the generated summaries along with the input sets of triples and the actual Wikipedia summaries in the resultant `.csv` file. The `.csv` file will by default be saved in the location of the pre-trained model.

For all possible alteration in the parameters of the above files, please consult their corresponding comment sections.

## KenLM
The `KenLM` directory contains all the required code in order to train an $n$-gram Kneser-Ney language model. The code is based on the [KenLM Language Model Toolkit](https://kheafield.com/code/kenlm/). The binary files that reside in the `./kenlm/build/` directory have been compiled using [Boost](http://www.boost.org/) on a machine running Ubuntu 16.04 (x86_64 Linux 4.4.0-98-generic). In case you wish to experiment with this baseline on a different OS, you need to download and compile the original package according to the instructions at [https://kheafield.com/code/kenlm/](https://kheafield.com/code/kenlm/).

The following Python packages should also be installed in your machine: (i) `numpy`, (ii) `pandas`, and (iii) `kenlm`. The latter can be installed by running: `pip install https://github.com/kpu/kenlm/archive/master.zip` (i.e. [https://github.com/kpu/kenlm](https://github.com/kpu/kenlm)).

* In a Unix shell environment, run: `sh train.sh` in order to train a $5$-gram Kneser-Ney language model. The trained model will be saved in the `./KenLM/` directory with the `.klm` extension (e.g. `D1.surf_form_tuples.model.klm` or `D2.surf_form_tuples.model.klm`).
* Execute the Python script `sample.py` in order to sample the most probable summary templates. The summaries are sampled using beam-search. The most probable templates will be saved in a `pickle` file (e.g. `D1.surf_form_tuples.templates.p` or `D2.surf_form_tuples.templates.p`) in the `./KenLM/templates/` directory.
* Run the Python script `process-templates.py` in order to post-process the templates according to each input set of triples from the test or validation set of the selected dataset. The script replaces the `<item>` tokens along with any potential property-type placeholders according to the triples of the input set. The generated `.csv` file with all the generated summaries along with their input sets of triples is saved in the `./KenLM/templates/` directory.

In the default scenario, the model trains on D1 and samples summaries for the sets of triples that have been allocated to the test set. In case you wish to run the files (i.e. `train.sh`, `sample.py` and `process-templates.py`) in a different setup, you can alter them following the guidelines in each file's comment sections. 

## License
This project is licensed under the terms of the Apache 2.0 License.