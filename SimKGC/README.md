# SimKGC Training and Dumping Embeddings

In this folder, we provide the baseline code for training SimKGC and dumping embeddings to use in ensembling. The processed datasets can be found in the respective dataset directories inside the `data` folder.

Most of the code is taken from the code repository of the following paper:

Liang Wang, Wei Zhao, Zhuoyu Wei, Jingming Liu,  "SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models", ACL 2022.

This code is available at:
https://github.com/intfloat/SimKGC

## Step 1: Preprocessing the Datasets
The datasets provided have already been processed. However, if this step has to be repeated, please run the following command:

```
bash scripts/preprocess.sh DATASET
```

where DATASET is one of `WN18RR`, `FB15k237` (for `FB15k-237`), `CodexM` (for `CoDex-M`) and `YAGO3-10`. 

## Step 2: Training SimKGC
Config files have already been provided for each dataset in the `scripts` folder. To run training, use the following command.

```
OUTPUT_DIR=CHECKPOINT_PATH bash scripts/train_DATASET.sh
```

where DATASET is one of `wn` (for `WN18RR`), `fb` (for `FB15k-237`), `codex` (for `CoDex-M`) and `yago` (for `YAGO3-10`). CHECKPOINT_PATH is the path where the trained model will be saved. 

## Step 3: Dumping SimKGC Embeddings
Finally, we dump embeddings for all possible `(h,r)` and `t` to disk, to use in the ensembling code. To do this, run the following command.

```
python dump_embs.py --task DATASET --is-test --eval-model-path CHECKPOINT_PATH/model_last.mdl --train-path data/DATASET/train.txt.json --valid-path data/DATASET/test.txt.json
```
where DATASET is one of `WN18RR`, `FB15k237` (for `FB15k-237`), `CodexM` (for `CoDex-M`) and `YAGO3-10`. CHECKPOINT_PATH is where the model was previously saved. This will create a folder `DATASET_Vectors` containing the SimKGC embeddings. Inside the folder, `SimKGC_t_rep.pkl` contains all tail embeddings. `SimKGC_h_REL_rep.pkl` contains embeddings for all heads given relation REL. 


## Requirements
- HuggingFace-Hub 0.11.1
- Tokenizers 0.10.3
- Transformers 4.15.0
- Joblib 1.2.0
- SciKit-Learn 1.0.2
- Python 3.7.13
- PyYAML 6.0
- Torch 1.6.0
- NumPy
- SciPy

