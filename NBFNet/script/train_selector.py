import os
import sys
import math
import pprint

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
from integrate import get_WN_mappings
from SimKGC_vectors import SimKGC
import pickle
from selector import Selector
from embedding import RotatE, ComplEx
# from info_nce import InfoNCE

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
import json


separator = ">" * 30
line = "-" * 30


def load_model(cfg, model, device):
    world_size = util.get_world_size()
    rank = util.get_rank()
    if rank == 0:
        logger.warning("Load checkpoint")
    state = torch.load(cfg.output_dir + cfg.load, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


def train_and_validate(cfg, train_data, valid_data, model, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0

    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        # print(list(model.named_parameters()))
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)

            for batch in train_loader:
                # print(batch)
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred, _ = parallel_model(train_data, batch, (epoch+1)/cfg.train.num_epoch, hard_wts=None)
                # print(pred)

                # target = torch.zeros_like(pred)
                # target[:, 0] = cfg.task.adversarial_temperature

                margin_target = torch.zeros([pred.size(dim=0)], dtype=torch.int64).to(batch.device)
                margin_loss = nn.MultiMarginLoss(margin=cfg.task.adversarial_temperature)
                margin_loss = margin_loss(pred, margin_target)

                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()
                # Codex Loss
                if (cfg.model.loss == 'sum'):
                    loss += margin_loss
                elif (cfg.model.loss == 'margin'):
                    loss = margin_loss

                # loss = nn.KLDivLoss(reduction="batchmean")
                # pred = F.log_softmax(pred, dim=1)
                # loss = loss(pred, target)

                # loss = nn.MSELoss()
                # loss = loss(pred, target)

                # loss = InfoNCE(negative_mode='paired')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, valid_data, model, wts=None, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    util.synchronize()


@torch.no_grad()
def test(cfg, test_data, model, wts=None, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    h_rankings = []
    t_rankings = []
    num_negatives = []
    nums = []
    h_types = []
    t_types = []

    if (cfg.ranklist_path != "None"):
        fi = open(cfg.ranklist_path, 'w')

    if (cfg.weight_path != "None"):
        fi = open(cfg.weight_path, 'w')
    
    if (cfg.feature_path != "None"):
        fi = open(cfg.feature_path, 'w')

    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)

        t_pred, t_weight = model(test_data, t_batch, hard_wts=wts)
        h_pred, h_weight = model(test_data, h_batch, hard_wts=wts)

        # h_max = torch.topk(h_pred, k=100, dim=1).indices
        # for _ in h_max.tolist():
        #     h_set = set()
        #     for idx in _:
        #         h_set.add(type_ent[idx])
        #     h_types.append(len(h_set))
        
        # t_max = torch.topk(t_pred, k=100, dim=1).indices
        # for _ in t_max.tolist():
        #     t_set = set()
        #     for idx in _:
        #         t_set.add(type_ent[idx])
        #     t_types.append(len(t_set))

        # h_index, t_index, r_index = batch.unbind(-1)
        # with open('~/SimKGC_Sample.txt', 'w') as fi:
        #     h_index = h_index.tolist()
        #     r_index = r_index.tolist()
        #     t_index = t_index.tolist()
        #     t_feat = t_pred.tolist()
        #     for _ in range(len(h_index)):
        #         fi.write(f'{t_index[_]}\n')
        #         fi.write(f'{" ".join([str(_) for _ in t_feat[_]])}\n')
        # break
                
        h_index, t_index, r_index = batch.unbind(-1)
        if (cfg.feature_path != "None"):
            h_index = h_index.tolist()
            r_index = r_index.tolist()
            t_index = t_index.tolist()
            t_feat = t_pred.tolist()
            h_feat = h_pred.tolist()
            for _ in range(len(h_index)):
                fi.write(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{" ".join([str(_) for _ in t_feat[_]])}\n')
                print(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{" ".join([str(_) for _ in t_feat[_]])}\n')
                fi.write(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{" ".join([str(_) for _ in h_feat[_]])}\n')
                print(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{" ".join([str(_) for _ in h_feat[_]])}\n')

        if (not cfg.model.get_feat):
            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            if (cfg.ranklist_path != "None"):
                h_index = h_index.tolist()
                r_index = r_index.tolist()
                t_index = t_index.tolist()
                for _ in range(len(h_index)):
                    fi.write(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{t_ranking[_]}\n')
                    print(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{t_ranking[_]}\n')
                    fi.write(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{h_ranking[_]}\n')
                    print(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{h_ranking[_]}\n')
                
            if (cfg.weight_path != "None"):
                h_index = h_index.tolist()
                r_index = r_index.tolist()
                t_index = t_index.tolist()
                for _ in range(len(h_index)):
                    fi.write(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{t_weight[_]}\n')
                    print(f'{nbfentmap_rev[h_index[_]]}\t{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[t_index[_]]}\t{t_weight[_]}\n')
                    fi.write(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{h_weight[_]}\n')
                    print(f'{nbfentmap_rev[t_index[_]]}\t!{nbfrelmap_rev[r_index[_]]}\t{nbfentmap_rev[h_index[_]]}\t{h_weight[_]}\n')

            rankings += [t_ranking, h_ranking]
            h_rankings += [t_ranking]
            t_rankings += [h_ranking]
            num_negatives += [num_t_negative, num_h_negative]
    
    if (cfg.model.get_feat):
        return None

    ranking = torch.cat(rankings)
    h_ranking = torch.cat(h_rankings)
    t_ranking = torch.cat(t_rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    all_h_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_h_size[rank] = len(h_ranking)
    if world_size > 1:
        dist.all_reduce(all_h_size, op=dist.ReduceOp.SUM)
    all_t_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_t_size[rank] = len(t_ranking)
    if world_size > 1:
        dist.all_reduce(all_t_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    cum_h_size = all_h_size.cumsum(0)
    cum_t_size = all_t_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_h_ranking = torch.zeros(all_h_size.sum(), dtype=torch.long, device=device)
    all_h_ranking[cum_h_size[rank] - all_h_size[rank]: cum_h_size[rank]] = h_ranking
    all_t_ranking = torch.zeros(all_t_size.sum(), dtype=torch.long, device=device)
    all_t_ranking[cum_t_size[rank] - all_t_size[rank]: cum_t_size[rank]] = t_ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_h_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_t_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        logger.warning("Positive Examples")
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_h_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_h_ranking.float()).mean()
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
                    score = (all_h_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))

    if rank == 0:
        logger.warning("Negative Examples")
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_t_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_t_ranking.float()).mean()
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
                    score = (all_t_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))

    if rank == 0:
        logger.warning("All Examples")
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
    nbf2siment, nbf2simrel, nbfentmap, nbfrelmap = get_WN_mappings(cfg.dataset["class"], cfg.dataset["root"])
    nbf2rnnent, nbf2rnnrel, nbfentmap, nbfrelmap = get_WN_mappings(cfg.dataset["class"], cfg.dataset["root"], rotate = True)
    nbfentmap_rev = {nbfentmap[_]:_ for _ in nbfentmap}
    nbfrelmap_rev = {nbfrelmap[_]:_ for _ in nbfrelmap}
    dim = 0
    if (cfg.model.need_sim):
        dim += 2
        simkgc = SimKGC(cfg.dataset["class"], cfg.simkgc, nbf2siment, nbf2simrel)
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    device = util.get_device(cfg)
    nbf = util.build_model(cfg)
    if (cfg.load != "None"):
        dim += 2
        load_model(cfg, nbf, device)
        nbf.eval()
    simkgc = None
    rotate = None
    complex = None
    if (cfg.model.need_rotate):
        dim += 2
        rotate = RotatE(cfg.rotate, nbf2rnnent, nbf2rnnrel)
    if (cfg.model.need_complex):
        dim += 2
        complex = ComplEx(cfg.complex, nbfentmap, nbfrelmap, cfg.model.div)
    model = Selector(cfg, nbf, simkgc, rotate, complex, 2, [cfg.model.ensemble_hidden_dim, cfg.model.ensemble_hidden_dim], dim, False)
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

    if (cfg.train_model):
        train_and_validate(cfg, valid_data, test_data, model, filtered_data=filtered_data)
    else:
        test(cfg, test_data, model, wts=None, filtered_data=filtered_data)
    
