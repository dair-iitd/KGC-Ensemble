import os
import sys
import ast
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR
from RNNLogic_integrate import get_WN_mappings

from nbfnet import models, datasets


logger = logging.getLogger(__file__)
nbf2rnnent, nbf2rnnrel, _, _ = get_WN_mappings()
rnn2nbfent = {nbf2rnnent[_]:_ for _ in nbf2rnnent}
rnn2nbfrel = {nbf2rnnrel[_]:_ for _ in nbf2rnnrel}
NUM_RELATIONS = 11

def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    cls = cfg.dataset.pop("class")
    if cls == "FB15k-237":
        dataset = RelLinkPredDataset(name=cls, **cfg.dataset)
        data = dataset.data
        train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type)
        valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                          target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type)
        test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                         target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type)
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    elif cls == "WN18RR":
        dataset = WordNet18RR(**cfg.dataset)
        # extra_triples_index = [[], []]
        # extra_triples_type = []
        # train_extra_triples_index = [[], []]
        # train_extra_triples_type = []
        # check = set()
        # with open(OTHER_DATA, 'r') as fi:
        #     for line in fi:
        #         line = [int(_) for _ in line.strip().split('\t')[:-1]]
        #         h = rnn2nbfent[line[0]]
        #         r = rnn2nbfrel[line[1]]
        #         t = rnn2nbfent[line[2]]
        #         if (h,r,t) not in check:
        #             extra_triples_index[0].append(h)
        #             extra_triples_index[1].append(t)
        #             extra_triples_type.append(r)
        #             check.add(tuple([h,r,t]))
        #             if (r < NUM_RELATIONS):
        #                 train_extra_triples_index[0].append(h)
        #                 train_extra_triples_index[1].append(t)
        #                 train_extra_triples_type.append(r)
        #         if (t,(r+NUM_RELATIONS)%(2*NUM_RELATIONS),h) not in check:
        #             extra_triples_index[0].append(t)
        #             extra_triples_index[1].append(h)
        #             extra_triples_type.append((r+NUM_RELATIONS)%(2*NUM_RELATIONS))
        #             check.add(tuple([t,(r+NUM_RELATIONS)%(2*NUM_RELATIONS),h]))
        #             if (r >= NUM_RELATIONS):
        #                 train_extra_triples_index[0].append(t)
        #                 train_extra_triples_index[1].append(h)
        #                 train_extra_triples_type.append((r+NUM_RELATIONS)%(2*NUM_RELATIONS))

        # print('Added Extra Data')
        # extra_triples_index = torch.IntTensor(extra_triples_index)
        # extra_triples_type = torch.IntTensor(extra_triples_type)
        # train_extra_triples_index = torch.IntTensor(train_extra_triples_index)
        # train_extra_triples_type = torch.IntTensor(train_extra_triples_type)

        # convert wn18rr into the same format as fb15k-237
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]
        # edge_index = torch.cat([edge_index, edge_index.flip(0), extra_triples_index], dim=-1)
        # edge_type = torch.cat([edge_type, edge_type + num_relations, extra_triples_type])
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        edge_type = torch.cat([edge_type, edge_type + num_relations])
        print(edge_index.shape)
        print(edge_type.shape)
        train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.train_mask],
                          target_edge_type=data.edge_type[data.train_mask])
        valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                          target_edge_index=data.edge_index[:, data.val_mask],
                          target_edge_type=data.edge_type[data.val_mask])
        test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                         target_edge_index=data.edge_index[:, data.test_mask],
                         target_edge_type=data.edge_type[data.test_mask])
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
        dataset.num_relations = num_relations * 2
    elif cls.startswith("Ind"):
        dataset = datasets.IndRelLinkPredDataset(name=cls[3:], **cfg.dataset)
    elif cls in ['UMLS', 'Kinship', 'CodexM', 'YAGO3-10']:
        dataset = WordNet18RR(root='~/datasets/knowledge_graphs/WN18RR/')
        root = cfg.dataset.pop("root")
        entities = {}
        with open(root + 'entities.dict', 'r') as fi:
            for line in fi:
                line = line.strip().split('\t')
                entities[line[1]] = int(line[0])
        relations = {}
        with open(root + 'relations.dict', 'r') as fi:
            for line in fi:
                line = line.strip().split('\t')
                relations[line[1]] = int(line[0])
        edge_index = [[], []]
        edge_type = []
        train_edge_index = [[], []]
        train_edge_type = []
        val_edge_index = [[], []]
        val_edge_type = []
        test_edge_index = [[], []]
        test_edge_type = []
        NUM_NODES = len(entities)
        NUM_RELATIONS = len(relations)//2
        with open(root + 'train.txt', 'r') as fi:
            for line in fi:
                line = line.strip().split('\t')
                h = entities[line[0]]
                r = relations[line[1]]
                t = entities[line[2]]
                if (r < NUM_RELATIONS):
                    train_edge_index[0].append(h)
                    train_edge_index[1].append(t)
                    train_edge_type.append(r)
                edge_index[0].append(h)
                edge_index[1].append(t)
                edge_type.append(r)
        with open(root + 'valid.txt', 'r') as fi:
            for line in fi:
                line = line.strip().split('\t')
                h = entities[line[0]]
                r = relations[line[1]]
                t = entities[line[2]]
                if (r < NUM_RELATIONS):
                    val_edge_index[0].append(h)
                    val_edge_index[1].append(t)
                    val_edge_type.append(r)
        with open(root + 'test.txt', 'r') as fi:
            for line in fi:
                line = line.strip().split('\t')
                h = entities[line[0]]
                r = relations[line[1]]
                t = entities[line[2]]
                if (r < NUM_RELATIONS):
                    test_edge_index[0].append(h)
                    test_edge_index[1].append(t)
                    test_edge_type.append(r)
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        train_edge_index = torch.LongTensor(train_edge_index)
        train_edge_type = torch.LongTensor(train_edge_type)
        val_edge_index = torch.LongTensor(val_edge_index)
        val_edge_type = torch.LongTensor(val_edge_type)
        test_edge_index = torch.LongTensor(test_edge_index)
        test_edge_type = torch.LongTensor(test_edge_type)
        print(edge_index.shape)
        print(edge_type.shape)
        train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=NUM_NODES,
                          target_edge_index=train_edge_index,
                          target_edge_type=train_edge_type)
        valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=NUM_NODES,
                          target_edge_index=val_edge_index,
                          target_edge_type=val_edge_type)
        test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=NUM_NODES,
                         target_edge_index=test_edge_index,
                         target_edge_type=test_edge_type)
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
        dataset.num_relations = NUM_RELATIONS * 2
    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    if get_rank() == 0:
        logger.warning("%s dataset" % cls)
        logger.warning("#train: %d, #valid: %d, #test: %d" %
                       (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                        dataset[2].target_edge_index.shape[1]))

    return dataset


def build_model(cfg):
    cls = cfg.model.pop("class")
    if cls == "NBFNet":
        model = models.NBFNet(**cfg.model)
    else:
        raise ValueError("Unknown model `%s`" % cls)
    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    return model
