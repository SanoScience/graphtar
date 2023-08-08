# GraphTar: applying word2vec and graph neural networks to miRNA target prediction
## Abstract
### Background
MicroRNAs (miRNAs) are short, non-coding RNA molecules that regulate gene expression by binding to specific mRNAs, inhibiting their translation. They play a critical role in regulating various biological processes and are implicated in many diseases, including cardiovascular, oncological, gastrointestinal diseases, and viral infections. Computational methods that can identify potential miRNA-mRNA interactions from raw data use one-dimensional miRNA-mRNA duplex representations and simple sequence encoding techniques, which may limit their performance.

### Results
We have developed GraphTar, a new target prediction method that uses a novel graph-based representation to reflect the spatial structure of the miRNA-mRNA duplex. Unlike existing approaches, we use the word2vec method to accurately encode RNA sequence information. In conjunction with a novel, word2vec-based encoding method, we use a graph neural network classifier that can accurately predict miRNA-mRNA interactions based on graph representation learning. As part of a comparative study, we evaluate three different node embedding approaches within the GraphTar framework and compare them with other state-of-the-art target prediction methods. The results show that the proposed method achieves similar performance to the best methods in the field and outperforms them on one of the datasets.

### Conclusions
In this study, a novel miRNA target prediction approach called GraphTar is introduced. Results show that GraphTar is as effective as existing methods and even outperforms them in some cases, opening new avenues for further research. However, the expansion of available datasets is critical for advancing the field towards real-world applications.
## Repository structure
The repository contains all the code used in the experiments. Specifically, it contains:
- reproduction steps, with all links to data needed to reproduce the experiments, provided by the [GraphTar sharepoint directory](https://sanoscience.sharepoint.com/:f:/s/graphtar/Eh2aeNfcstZPtQ5sTnspgWoBZc9eryJvomJ-vDmri4BP2w?e=4x1vdi).
- dataset integration
- model configurations
- [NeptuneAI](https://neptune.ai/?utm_source=googleads&utm_medium=googleads&utm_campaign=[SG][HI][brand][rsa][all]&utm_term=neptune%20ai&gclid=Cj0KCQjwnf-kBhCnARIsAFlg49000HIobJdPqbh6uM67AFWssPBLd74m5mFSLZgRatRrBEVPc-fmQ6EaAgHCEALw_wcB) integration, for experiment tracking
- GraphTar method
- state-of-the-art algorithms reproduced for comparison with GraphTar:
  - [miRAW: A deep learning-based approach to predict microRNA targets by analyzing whole microRNA transcripts](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006185)
  - [DeepMirTar: a deep-learning approach for predicting human miRNA targets](https://academic.oup.com/bioinformatics/article/34/22/3781/5026656)
  - [miTAR: a hybrid deep learning-based approach for predicting miRNA targets](https://link.springer.com/article/10.1186/s12859-021-04026-6)
- analysis scripts
- experiment scripts

The directory structure is provided below, with short explanation for the most notable files and directories:
```
📦gnn-target-prediction
 ┣ 📂analysis
 ┃ ┗ 📂results_analysis
 ┃ ┃ ┗ 📜results_analysis.ipynb <- notebook used to compute metric scores for the results section.
 ┣ 📂data_modules
 ┃ ┣ 📂configs <- directory with configuration per method-dataset combination. Configs specify data paths, transforms used, as well as train/test splits.
 ┃ ┃ ┣ 📜dataset_config.py
 ┃ ┃ ┣ 📜deepmirtar_config.json
 ┃ ┃ ┣ 📜deepmirtar_in_config.json
 ┃ ┃ ┣ 📜graphtar_config_deepmirtar.json
 ┃ ┃ ┣ 📜graphtar_config_deepmirtar_w2v.json
 ┃ ┃ ┣ 📜graphtar_config_miraw.json
 ┃ ┃ ┣ 📜graphtar_config_miraw_w2v.json
 ┃ ┃ ┣ 📜graphtar_config_mirtarraw.json
 ┃ ┃ ┣ 📜graphtar_config_mirtarraw_w2v.json
 ┃ ┃ ┣ 📜miraw_config.json
 ┃ ┃ ┣ 📜miraw_in_config.json
 ┃ ┃ ┣ 📜mirtarraw_config.json
 ┃ ┃ ┗ 📜mitar_config.json
 ┃ ┣ 📂datasets
 ┃ ┃ ┣ 📂transforms <- all data transforms used for preprocessing
 ┃ ┃ ┃ ┣ 📂word2vec
 ┃ ┃ ┃ ┃ ┣ 📂models
 ┃ ┃ ┃ ┃ ┃ ┗ 📜generate_w2v_models.ipynb <- notebook used to generate word2vec models for every dataset-split combination
 ┃ ┃ ┃ ┃ ┗ 📜to_word2vec_embedding.py
 ┃ ┃ ┃ ┣ 📜json_serializable.py
 ┃ ┃ ┃ ┣ 📜merge.py
 ┃ ┃ ┃ ┣ 📜pad.py
 ┃ ┃ ┃ ┣ 📜to_one_hot.py
 ┃ ┃ ┃ ┗ 📜to_tensor.py
 ┃ ┃ ┣ 📜interaction_dataset.py <- Pytorch dataset implementation for interaction data used in reproduced methods
 ┃ ┃ ┗ 📜interaction_graph_dataset.py <- Pytorch dataset implementation for interaction data used in GraphTar (graph-based representation)
 ┃ ┣ 📜graph_interaction_data_module.py <- Pytorch Lightning data module for GraphTar
 ┃ ┗ 📜interaction_data_module.py <- Pytorch Lightning data module for reproduced methods
 ┣ 📂experiments <- all scripts used for training and evaluation in the reported experiments. For all reproduced methods, the structure follows the same pattern (see deepmirtar directory for details)
 ┃ ┣ 📂deepmirtar
 ┃ ┃ ┣ 📜ann.py <- ANN training script
 ┃ ┃ ┣ 📜ann_test.py <- ANN evaluation script
 ┃ ┃ ┣ 📜run_ann_training_tunning.sh <- SLURM script used for simple tests of ANN training logic
 ┃ ┃ ┣ 📜run_array_ann_training_seed.sh <- SLURM script used to train final ANN part for every seed (after finding the best hyperparameters)
 ┃ ┃ ┣ 📜run_array_ann_training_tunning.sh <- SLURM script used to train ANN part in the hyperparameters tunning phase
 ┃ ┃ ┣ 📜run_array_autoencoder_training_seed.sh <- SLURM script used to train final SdA models for every seed (after finding the best hyperparameters)
 ┃ ┃ ┣ 📜run_array_autoencoder_training_tunning.sh <- SLURM script used to train SdA part in the hyperparameters tunning phase
 ┃ ┃ ┣ 📜run_autoencoder_training_tunning.sh <- SLURM script used for simple tests of SdA training logic
 ┃ ┃ ┗ 📜sd_autoencoder.py <- SdA training script
 ┃ ┣ 📂graphtar <- the structure follows the same pattern as for other methods, but files with 'w2v' in the file name use word2vec encoding, whereas files without 'w2v' use one-hot encoding
 ┃ ┃ ┣ 📜gnn.py
 ┃ ┃ ┣ 📜gnn_w2v.py
 ┃ ┃ ┣ 📜gnn_w2v_test.py
 ┃ ┃ ┣ 📜run_array_graphtar_net_training_seed.sh
 ┃ ┃ ┣ 📜run_array_graphtar_net_training_seed_w2v.sh
 ┃ ┃ ┣ 📜run_array_graphtar_net_training_tunning.sh
 ┃ ┃ ┣ 📜run_array_graphtar_net_training_tunning_w2v.sh
 ┃ ┃ ┣ 📜run_graphtar_net_training_tunning.sh
 ┃ ┃ ┣ 📜run_graphtar_net_training_tunning_local_w2v.sh
 ┃ ┃ ┗ 📜run_graphtar_net_training_tunning_w2v.sh
 ┃ ┣ 📂miraw
 ┃ ┃ ┣ 📜ann.py
 ┃ ┃ ┣ 📜ann_test.py
 ┃ ┃ ┣ 📜autoencoder.py
 ┃ ┃ ┣ 📜run_ann_training_tunning.sh
 ┃ ┃ ┣ 📜run_array_ann_training_seed.sh
 ┃ ┃ ┣ 📜run_array_ann_training_tunning.sh
 ┃ ┃ ┣ 📜run_array_autoencoder_training_seed.sh
 ┃ ┃ ┣ 📜run_array_autoencoder_training_tunning.sh
 ┃ ┃ ┗ 📜run_autoencoder_training_tunning.sh
 ┃ ┗ 📂mitar
 ┃ ┃ ┣ 📜mitar_net.py
 ┃ ┃ ┣ 📜mitar_net_test.py
 ┃ ┃ ┣ 📜run_array_mitar_net_training_seed.sh
 ┃ ┃ ┣ 📜run_array_mitar_net_training_tunning.sh
 ┃ ┃ ┗ 📜run_mitar_net_training_tunning.sh
 ┣ 📂lightning_modules <- Pytorch Lightning modules for all methods
 ┃ ┣ 📂deepmirtar
 ┃ ┃ ┣ 📜ann.py
 ┃ ┃ ┗ 📜sd_autoencoder.py
 ┃ ┣ 📂graphtar
 ┃ ┃ ┗ 📜gnn.py
 ┃ ┣ 📂miraw
 ┃ ┃ ┣ 📜ann.py
 ┃ ┃ ┗ 📜autoencoder.py
 ┃ ┣ 📂mitar
 ┃ ┃ ┗ 📜mitar_net.py
 ┃ ┗ 📂models <- Pytorch models for all methods. Note, that for DeepMirTar and miRAW, models are divided into ann and autoencoder parts.
 ┃ ┃ ┣ 📂deepmirtar
 ┃ ┃ ┃ ┣ 📜ann.py
 ┃ ┃ ┃ ┗ 📜denoising_autoencoder.py
 ┃ ┃ ┣ 📂graphtar
 ┃ ┃ ┃ ┗ 📜gnn.py
 ┃ ┃ ┣ 📂miraw
 ┃ ┃ ┃ ┣ 📜ann.py
 ┃ ┃ ┃ ┗ 📜autoencoder.py
 ┃ ┃ ┗ 📂mitar
 ┗ ┗ ┗ ┗ 📜mitar_net.py
```

## Dataset
Dataset used in this study is available through [GraphTar sharepoint directory](https://sanoscience.sharepoint.com/:f:/s/graphtar/ErgU2YfS-OdLle6J7njRMOEBZ1kxjUxWKAqCeKna6_dxKQ?e=rf5Pi2).
The ```.csv``` files should be placed in the ```/data``` directory in the project's root.
## Models
### Word2vec models
For GraphTar models, the data was first encoded with the use of word2vec models pretrained on the training data. For each dataset and split, a separate model was trained. The trained word2vec models are available through [GraphTar sharepoint directory](https://sanoscience.sharepoint.com/:f:/s/graphtar/EhpQYk87MVZEpYD5n0SoE-QBpt1BWz3oJ-DPm1JOsDoOXg?e=ckWhvf).
These models should be placed in ```data_modules/datasets/transforms/word2vec/models/``` directory, so that they are correctly used by training and evaluation scripts.
### Target prediction models
Checkpoints for the trained traget prediction models are available through [GraphTar sharepoint directory](https://sanoscience.sharepoint.com/:f:/s/graphtar/EuwSMNa8qfRMke4hHVotMUgBLcr21ICuIxkf1_4zSJccsg?e=aKZ8gp).
The checkpoints should be placed in ```experiments/[METHOD]/models```, where [METHOD] is the respective target prediction method. E.g. for GraphTar, the GraphTar models should be placed in ```experiments/graphtar/models``` directory.

## Results
To calculate metric scores, we have saved data frames with predictions for all target prediction method-dataset-data split combinations to dataframes (more on how they are used in the last section). If one wants to skip the prediction generation step, the data frames are available through [GraphTar sharepoint directory](https://sanoscience.sharepoint.com/:f:/s/graphtar/ErTJMZOksWBJsmko3kFOz-kBFlYdrRKb-6BtG6kUYudfOA?e=j3DlA0). They should be placed in ```experiments/[METHOD]/results``` directory, depending on the target prediction method. E.g. for GraphTar, the respective dataframes should be placed in ```experiments/graphtar/results``` directory.

## Experiments reproduction
### Environment setup
With Python 3.8.10 as base interpreter, to reproduce the experiments, one has to first set up the virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Running models training
#### Training word2vec models
Pre-training Word2vec models are required to run GraphTar method with word2vec-based sequence encoding. To generate trained models, the notebook ```data_modules/datasets/transforms/word2vec/models/generate_w2v_models.ipynb``` can be used. Opening the notebook in ```jupyter-lab``` and running all cells will train and save models for every split-dataset combination (in experiments we use 30 splits and 3 datasets).
#### Training target prediction methods
To train target prediction models, one can use scripts from the ```experiments/``` directory. Note, that to run the training successfully, one has to first download the data to the ```/data``` directory. Also, for GraphTar with word2vec based encoding, one needs to download or generate the models for every split and dataset, so that they are present in ```data_modules/datasets/transforms/word2vec/models/```
To run the GraphTar training locally, one can use the ```experiments/graphtar/run_graphtar_net_training_tunning_local_w2v.sh``` script:
```sh
cd experiments/graphtar
./run_graphtar_net_training_tunning_local_w2v.sh
```
In every bash script in the ```experiments/``` directory, the crucial two lines are:
```sh
# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/graphtar/gnn_w2v.py data_modules/configs/graphtar_config_deepmirtar_w2v.json GAT ADD 64 2 64 2 0.4 1234 0.001 128 1000 experiments/graphtar/models
```
The commented line indicates the parameters passed to the ```gnn_w2v.py``` training script. By modifying this line, one can change the hyperparameters (dropout rate, lr, batch size, epochs number) as well as architecture (in case of GraphTar - node embedding method, number of graph embedding layers, node embedding size, number of fully connected layers and fc layers size). One can also change the data split seed used.

To train a model for every data split seed used in the experiments, we used PLGrid computing resources through SLURM, by running an array job:
```sh
cd experiments/graphtar
sbatch run_array_graphtar_net_training_tunning_w2v.sh
```
Note that for every node embedding method, a different set of hyperparameters was used, and that you have to set them up manually before execution, within the ```run_array_graphtar_net_training_tunning_w2v.sh``` script.
### Running models evaluation
Note: for steps below, one has to download the data and the models first (see previous sections)!
We evaluated the models in two steps. First, for every model, dataset and split combination, we have generated a data frame in ```.csv``` format. In this data frame we stored the predictions of the model as well as ground truth. To generate the data frames, one can use scripts from the ```experiments/``` directory with ```_test``` suffix. E.g. for GraphTar:
```
cd experiments/graphtar
./run_graphtar_net_w2v_test.sh
```
the data frames will be generated to ```experiments/[METHOD]/results``` directory, depending on the target prediction method. For GraphTar, it will be ```experiments/graphtar/results``` directory.

After generating the data frames, we used a jupyter notebook to analyse the results. The notebook can be found in ```analysis/results_analysis/results_analysis.ipynb```. Running all cells of the notebook will provide metric scores for all target prediction methods and datasets. Note, that the provided results are averaged across data splits.


