import os
import sys
import math
import pprint
import json

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
# from script.RNNLogic_integrate import get_WN_mappings, get_index_dict, get_SimKGC_scores 


separator = ">" * 30
line = "-" * 30

# WN = False

def load_model(cfg, model):
    world_size = util.get_world_size()
    rank = util.get_rank()
    if rank == 0:
        logger.warning("Load checkpoint")
    state = torch.load(cfg.output_dir + cfg.load, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    rank_dict = {}
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()


        batch_size = t_batch.size(dim=0)
            
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    load_model(cfg, model)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)
