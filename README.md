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

## Data
## Models
## Experiments reproduction
### Environment setup
### Running models training
### Running models evaluation

