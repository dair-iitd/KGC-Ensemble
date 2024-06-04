import torch
from torch import nn, autograd
from SimKGC_vectors import Reachability
import json
import numpy as np

class Selector(nn.Module):
    def __init__(self, cfg, nbf, simkgc, rotate, complex, num_layers, hidden_dims, input_dim, train=False):
        super(Selector, self).__init__()

        self.nbf = nbf
        if (not train):
            self.nbf.requires_grad_(False)
        if (simkgc is not None):
            self.simkgc = simkgc
        if (rotate is not None):
            self.rotate = rotate
            self.rotate.requires_grad_(False)
            self.tot_rels = self.rotate.remb.size(0)

        if (complex is not None):
            self.complex = complex
            self.complex.requires_grad_(False)
        
        self.cfg = cfg

        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            mlp.append(layer)
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_dims[-1], 1))
        mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

        rotate_mlp = []
        rotate_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            rotate_mlp.append(layer)
            rotate_mlp.append(nn.ReLU())
        rotate_mlp.append(nn.Linear(hidden_dims[-1], 1))
        rotate_mlp.append(nn.ReLU())
        self.rotate_mlp = nn.Sequential(*rotate_mlp)

        simkgc_mlp = []
        simkgc_mlp.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(num_layers - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            if (cfg.model.init != 0.0):
                torch.nn.init.uniform_(layer.weight, 0.0, cfg.model.init)
            simkgc_mlp.append(layer)
            simkgc_mlp.append(nn.ReLU())
        simkgc_mlp.append(nn.Linear(hidden_dims[-1], 1))
        simkgc_mlp.append(nn.ReLU())
        self.simkgc_mlp = nn.Sequential(*simkgc_mlp)

    def get_rotate_scores(self, all_h, r_head):
        all_h = all_h[:, 0]
        r_head = r_head[:, 0]
        h_emb = self.rotate.eemb[all_h].squeeze()
        rule_emb = self.rotate.remb[r_head]
        rule_emb = self.rotate.project(rule_emb)
        h_emb = self.rotate.product(h_emb, rule_emb)
        t_emb = self.rotate.eemb
        h_emb = torch.unsqueeze(h_emb, 1)
        dists = self.rotate.diff(h_emb, t_emb).sum(dim=-1)
        return -dists
    
    def get_rotate_rep(self, all_h, r_head):
        all_h = all_h[:, 0]
        r_head = r_head[:, 0]
        rotate = self.rotate
        h_emb = rotate.eemb[all_h].squeeze()
        rule_emb = rotate.remb[r_head]
        rule_emb = rotate.project(rule_emb)
        h_emb = rotate.product(h_emb, rule_emb)
        return h_emb

    def get_features_and_normalize(self, score, t_index):
        pre_maxs = torch.amax(score, dim=1).unsqueeze(dim = -1)
        pre_mins = torch.amin(score, dim=1).unsqueeze(dim = -1)
        pre_diffs = pre_maxs - pre_mins
        score = (score - pre_mins)
        maxs = torch.amax(score, dim=1).unsqueeze(dim = -1)
        score = score/maxs
        var = torch.var(score, dim=1).unsqueeze(dim = -1)
        std = torch.std(score, dim=1).unsqueeze(dim = -1)
        means = torch.mean(score, dim=1).unsqueeze(dim = -1)
        mdiffs = 1 - means
        topk = torch.mean(torch.topk(score, dim=1, k=10, largest=True).values, dim=1).unsqueeze(dim = -1)
        topk_std = torch.std(torch.topk(score, dim=1, k=10, largest=True).values, dim=1).unsqueeze(dim = -1)
        prop = (torch.sum((score >= 0.8), axis = -1)/score.size(1)).unsqueeze(dim = -1)
        # index = torch.topk(score, k=10, dim=-1, largest=True).indices
        # add = torch.zeros(score.shape, device = batch.device)
        # add[:, index] = 0.1
        # score += add
        if self.training:
            score = torch.gather(score, -1, t_index)
        return score, [mdiffs, var]
        # return score, [mdiffs, std, 1 - prop]

    def forward(self, data, batch, edge_weight=None, hard_wts=None):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            data, mask = self.nbf.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        h_index, t_index, r_index = self.nbf.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        features = []

        if (self.cfg.model.need_nbf):
            nbf_output = self.nbf.bellmanford(data, h_index[:, 0], r_index[:, 0])
            feature = nbf_output["node_feature"]
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # feature = feature.gather(1, index)

            # probability logit for each tail node in the batch
            # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
            nbf_score = self.nbf.mlp(feature).squeeze(-1)
            nbf_score, feature = self.get_features_and_normalize(nbf_score, t_index)
            features += feature
        
        if (self.cfg.model.need_sim):
            simkgc_score = self.simkgc.get_h(h_index, r_index).to(h_index.device) 
            simkgc_score, feature = self.get_features_and_normalize(simkgc_score, t_index)
            features += feature

        if (self.cfg.model.need_rotate):
            rotate_score = self.get_rotate_scores(h_index, r_index)
            rotate_score, feature = self.get_features_and_normalize(rotate_score, t_index)
            features += feature

        if (self.cfg.model.need_complex):
            complex_score = self.complex.get_score(h_index[:, 0], r_index[:, 0])
            complex_score, feature = self.get_features_and_normalize(complex_score, t_index)
            features += feature

        if (self.cfg.model.method == 'rerank'):
            if (self.cfg.model.rerank == 'forward'):
                vals = torch.topk(nbf_score, 100, sorted=True).values
                vals = vals[:, -1].unsqueeze(-1)
                mask = nbf_score >= vals

                return nbf_score + mask*1000.0*(1  + simkgc_score), None
            else:
                vals = torch.topk(simkgc_score, 100, sorted=True).values
                vals = vals[:, -1].unsqueeze(-1)
                mask = simkgc_score >= vals

                return simkgc_score + mask*1000.0*(1  + nbf_score), None

        if (self.cfg.model.method == 'ensemble'):
            mlp_in = torch.cat(tuple(features), dim = 1)

            if (self.cfg.model.get_feat):
                return mlp_in, None

            if (self.cfg.model.ensemble_nbf):
                weight = self.mlp(mlp_in.to(h_index.device))
                # print(weight[:4, :])

            if (self.cfg.model.ensemble_atomic):
                rotate_weight = self.rotate_mlp(mlp_in.to(h_index.device))
                # print(rotate_weight[:4, :])

            if (self.cfg.model.ensemble_sim):
                simkgc_weight = self.simkgc_mlp(mlp_in.to(h_index.device))
                # print(simkgc_weight[:4, :])

            result = 0
            if (self.cfg.model.need_nbf):
                if (self.cfg.model.ensemble_nbf):
                    result += weight*nbf_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[0]
                    result += wt*nbf_score
            if (self.cfg.model.need_sim):
                if (self.cfg.model.ensemble_sim):
                    result += simkgc_weight*simkgc_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[1]
                    # wt = torch.FloatTensor(simkgc_score.shape).uniform_(2.0, 3.0).to(r_index.device)
                    result += wt*simkgc_score
            if (self.cfg.model.need_rotate):
                if (self.cfg.model.ensemble_atomic):
                    result += rotate_weight*rotate_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[2]
                    # wt = torch.FloatTensor(simkgc_score.shape).uniform_(0.5, 1.0).to(r_index.device)
                    result += wt*rotate_score
            if (self.cfg.model.need_complex):
                if (self.cfg.model.ensemble_atomic):
                    result += rotate_weight*complex_score
                else:
                    wt = 1.0
                    if (hard_wts is not None):
                        wt = hard_wts[3]
                    result += wt*complex_score
            if (self.cfg.weight_path is not None):
                return result, nbf_weight
            else:
                return result, None