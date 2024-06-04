import os
import json
import tqdm
import torch
import sys
from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict
from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from logger_config import logger
import json
import pickle
import numpy as np

entity_dict = get_entity_dict()
all_triplet_dict = get_all_triplet_dict()


if __name__=='__main__':
    DATASET = args.task
    if (DATASET not in ['WN18RR', 'YAGO3-10', 'FB15k237', 'CodexM']):
        print('Dataset not implemented!')
        sys.exit(0)
    if (DATASET == 'WN18RR'):
        NUM_RELATIONS = 22
    elif (DATASET == 'FB15k237'):
        NUM_RELATIONS = 474
    elif (DATASET == 'CodexM'):
        NUM_RELATIONS = 142
    elif (DATASET == 'YAGO3-10'):
        NUM_RELATIONS = 74
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)
    pickle.dump(entity_tensor, open(f'{DATASET}_Vectors/SimKGC_t_rep.pkl', 'wb'))
    print('Dumped Tails')
    relations = []
    with open(f'data/{DATASET}/relations.dict', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            relations.append(line[1])
    if (DATASET == 'YAGO3-10'):
        s_relations = {_:_ for _ in relations}
    else:
        s_relations = json.load(open(f'data/{DATASET}/relations.json', 'r'))
    for rel in range(NUM_RELATIONS):
        print(f'Computing for Relation {rel}')
        examples = []
        for entity in entity_dict.entity_exs:
            head_id = entity.entity_id
            if (rel >= NUM_RELATIONS//2):
                examples.append(Example(head_id, 'inverse '+s_relations[relations[rel - NUM_RELATIONS//2]], head_id))
            else:
                examples.append(Example(head_id, s_relations[relations[rel]], head_id))
        hr_tensor, _ = predictor.predict_by_examples(examples)
        pickle.dump(hr_tensor, open(f'{DATASET}_Vectors/SimKGC_h_{rel}_rep.pkl', 'wb'))
        