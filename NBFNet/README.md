# NBFNet Training and Ensemble Code

In this folder, we provide the baseline code for training NBFNet. We also provide the code used to run our ensemble training and analysis experiments. The datasets can be found in the respective dataset directories inside the `datasets/knowledge_graphs` folder.

Most of the code is taken from the code repository of the following paper:

Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, Jian Tang,  "Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction", NeurIPS 2021.

This code is available at:
https://github.com/KiddoZhu/NBFNet-PyG

## Step 1: Training NBFNet
The config files for training NBFNet have been provided in the `config` folder. To run NBFNet on single GPU, use the following command. 

```
python script/run.py -c config/DATASET_nbf.yaml --gpus [0]
```

To use multiple GPUs, use the following command. 

```
python -m torch.distributed.launch --nproc_per_node=K script/run.py -c config/DATASET_nbf.yaml --gpus [0,1,...K]
```

In the above commands, DATASET is one of `wn18rr` (for `WN18RR`), `fb15k237` (for `FB15k-237`), and `codex` (for `CoDex-M`). K is the number of GPUs on which training is to be run.

## Step 2: Running Ensemble Code
We use the same dataloader and framework as NBFNet to run our ensemble experiments. The config files for the different datasets are in the `config` folder. 

First, we explain some of the important options provided in the configuration:
- `load`: path to NBFNet checkpoint
- `rotate`: path to RotatE checkpoint
- `complex`: path to ComplEx checkpoint
- `simkgc`: parent folder of SimKGC vector folder
- `ranklist_path`: path where the ranklist of the model on test set will be saved, if required
- `weight_path`: path where the ensemble weight of the model on test set will be saved, if required
- `train_model`: set to train ensemble, otherwise fixed weights can be provided in line 425 of `script/train_selector.py` for evaluation on test set
- `need_nbf`: set to use NBFNet in the ensemble
- `need_sim`: set to use SimKGC in the ensemble
- `need_rotate`: set to use RotatE in the ensemble
- `need_complex`: set to use ComplEx in the ensemble
- `ensemble_nbf`: set to make NBFNet's ensemble weight trainable
- `ensemble_sim`: set to make SimKGC's ensemble weight trainable
- `ensemble_atomic`: set to make RotatE/ComplEx's ensemble weight trainable
- `ensemble_hidden_dim`: hidden dimension of the ensemble weight MLP

We recommend using full paths whenever possible, as the original code switches between working directories during execution. 

The ensemble experiment can then be run using:

```
python script/train_selector.py -c config/DATASET.yaml --gpus [0]
```

where DATASET is one of `wn18rr` (for `WN18RR`), `fb15k237` (for `FB15k-237`), and `codex` (for `CoDex-M`).

## Other Model-Combination Techniques
All other techniques can be run from the same config files as our dynamic ensemble experiments.

### Static Ensembling
To run static ensembling, set `method` to 'ensemble'. Set `train_model` to no. Provide the fixed weights in line 425 of `script/train_selector.py` (the wts parameter) for evaluation on test set. The weights array has four elements, containing the NBFNet, SimKGC, RotatE and ComplEx ensemble weights in order. 

### Re-ranking
To run re-ranking, set `method` to 'rerank'. Setting `rerank` to forward will do NBFNet-SimKGC reranking. Otherwise, SimKGC-NBFNet reranking will be performed. 

## Requirements
- BrotliPy 0.7.0
- EasyDict 1.9
- Jinja2 3.1.2
- Joblib 1.2
- Ninja 1.11.0
- NumPy 1.21.6
- Pandas 1.3.5
- Python 3.7.13
- Torch 1.8.0
- Pytorch-Scatter 2.0.8
- Pytorch-Sparse 0.6.12
- Pytorch-Geometric 
- PyYaml 6.0
- SciKit-Learn 1.0.2
- SciPy 1.7.3
- Yaml 0.2.5
