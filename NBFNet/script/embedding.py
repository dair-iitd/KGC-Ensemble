import os
import torch
import numpy as np
import json

class RotatE(torch.nn.Module):
    def __init__(self, path, nbf2rnnent, nbf2rnnrel):
        super(RotatE, self).__init__()
        self.path = path

        self.re_index = [nbf2rnnent[_] for _ in range(len(nbf2rnnent))]
        self.re_rel = [nbf2rnnrel[_] for _ in range(len(nbf2rnnrel))]

        cfg_file = os.path.join(path, 'config.json')
        with open(cfg_file, 'r') as fi:
            cfg = json.load(fi)
        self.emb_dim = cfg['hidden_dim']
        self.gamma = cfg['gamma']
        self.range = (self.gamma + 2.0) / self.emb_dim
        self.num_entities = cfg['nentity']

        eemb_file = os.path.join(path, 'entity_embedding.npy')
        eemb = np.load(eemb_file)
        eemb = eemb[self.re_index, :]
        self.eemb = torch.nn.parameter.Parameter(torch.tensor(eemb))

        remb_file = os.path.join(path, 'relation_embedding.npy')
        remb = np.load(remb_file)
        remb = torch.tensor(remb)
        remb = torch.cat([remb, -remb], dim=0)
        remb = remb[self.re_rel, :]
        self.remb = torch.nn.parameter.Parameter(remb)

    def product(self, vec1, vec2):
        re_1, im_1 = torch.chunk(vec1, 2, dim=-1)
        re_2, im_2 = torch.chunk(vec2, 2, dim=-1)

        re_res = re_1 * re_2 - im_1 * im_2
        im_res = re_1 * im_2 + im_1 * re_2

        return torch.cat([re_res, im_res], dim=-1)

    def project(self, vec):
        pi = 3.141592653589793238462643383279
        vec = vec / (self.range / pi)

        re_r = torch.cos(vec)
        im_r = torch.sin(vec)

        return torch.cat([re_r, im_r], dim=-1)

    def diff(self, vec1, vec2):
        diff = vec1 - vec2
        re_diff, im_diff = torch.chunk(diff, 2, dim=-1)
        # diff = torch.stack([re_diff, im_diff], dim=0)
        # diff = diff.norm(dim=0)
        diff = torch.sqrt(re_diff**2 + im_diff**2)
        return diff

    def dist(self, all_h, all_r, all_t):
        h_emb = self.eemb.index_select(0, all_h).squeeze()
        r_emb = self.remb.index_select(0, all_r).squeeze()
        t_emb = self.eemb.index_select(0, all_t).squeeze()

        r_emb = self.project(r_emb)
        e_emb = self.product(h_emb, r_emb)
        dist = self.diff(e_emb, t_emb)

        return dist.sum(dim=-1)
    
    def forward(self, all_h, all_r):
        all_h_ = all_h.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_r_ = all_r.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_e_ = torch.tensor(list(range(self.num_entities)), dtype=torch.long, device=all_h.device).unsqueeze(0).expand(all_r.size(0), -1).reshape(-1)
        kge_score = self.gamma - self.dist(all_h_, all_r_, all_e_)
        kge_score = kge_score.view(-1, self.num_entities)
        return kge_score

class ComplEx(torch.nn.Module):
    def __init__(self, path, nbfentmap, nbfrelmap, div=True):
        super(ComplEx, self).__init__()
        self.path = path

        ckpt = os.path.join(path, 'best_valid_model.pt')
        ckpt = torch.load(ckpt)
        embs = ckpt["model_weights"]
        r_map = ckpt["relation_map"]
        e_map = ckpt["entity_map"]

        # print(r_map)

        # print(embs)

        # print(nbfentmap)

        nbf2complexent = {nbfentmap[_]:e_map[_] for _ in e_map if 'OOV' not in _}
        nbf2complexrel = {nbfrelmap[_]:r_map[_] for _ in r_map}

        self.embedding_dim = embs['E_im.weight'].size(1)
        self.rel_dim = embs['R_im.weight'].size(1)
        
        self.re_index = [nbf2complexent[_] if _ in nbf2complexent else (len(nbfentmap) - 1) for _ in range(len(nbfentmap))]
        self.re_rel = [nbf2complexrel[_] if _ in nbf2complexrel else (len(nbfrelmap) - 1) for _ in range(len(nbfrelmap))]

        pad = torch.zeros(len(nbfentmap) - len(nbf2complexent), self.embedding_dim).to(embs['E_im.weight'].device)
        pad_rel = torch.rand(len(nbfrelmap) - len(nbf2complexrel), self.rel_dim).to(embs['R_im.weight'].device)

        self.E_im = torch.cat([embs['E_im.weight'], pad], dim = 0)[self.re_index, :]
        self.R_im = torch.cat([embs['R_im.weight'], pad_rel], dim = 0)[self.re_rel, :]
        self.E_re = torch.cat([embs['E_re.weight'], pad], dim = 0)[self.re_index, :]
        self.R_re = torch.cat([embs['R_re.weight'], pad_rel], dim = 0)[self.re_rel, :]

        self.num_relations = len(nbfrelmap)
        if div:
            self.num_relations = self.num_relations//2
        self.num_entities = len(nbfentmap)
        self.ignore_ind = [_ for _ in range(len(nbfentmap)) if _ not in nbf2complexent]
        self.mask = torch.zeros(1, self.num_entities)
        self.mask[0, self.ignore_ind] = -10.0

        print(self.num_relations)

    def get_score(self, h, r):
        h_im = self.E_im[h, :]
        h_re = self.E_re[h, :]

        all_e_im = self.E_im.unsqueeze(0).view(-1, self.embedding_dim).transpose(0, 1)
        all_e_re = self.E_re.unsqueeze(0).view(-1, self.embedding_dim).transpose(0, 1)

        result = torch.zeros(h.size(0), self.num_entities).to(h.device)
        for _ in range(h.size(0)):
            h_curr = h[_]
            r_curr = r[_]

            if (r_curr >= self.num_relations):
                r_curr -= self.num_relations
                r_im = self.R_im[r_curr, :]
                r_re = self.R_re[r_curr, :]

                # print(h_im[_, :].shape)
                # print(all_e_im.shape)


                tmp1 = h_im[_, :]*r_re - h_re[_, :]*r_im
                tmp2 = h_im[_, :]*r_im + h_re[_, :]*r_re


                result[_] = tmp1.unsqueeze(0)@all_e_im + tmp2.unsqueeze(0)@all_e_re
            else:
                r_im = self.R_im[r_curr, :]
                r_re = self.R_re[r_curr, :]

                # print(h_im[_, :].shape)
                # print(all_e_im.shape)

                tmp1 = h_im[_, :]*r_re + h_re[_, :]*r_im
                tmp2 = h_re[_, :]*r_re - h_im[_, :]*r_im

                result[_] = tmp1.unsqueeze(0)@all_e_im + tmp2.unsqueeze(0)@all_e_re

        return result + self.mask.to(h.device)

    def get_score_check(self, h, walk):
        h_im = self.E_im[h, :]
        h_re = self.E_re[h, :]

        r_im = self.R_im[walk[0], :]
        r_re = self.R_re[walk[0], :]

        tmp1 = h_im*r_re + h_re*r_im
        tmp2 = h_re*r_re - h_im*r_im

        r_im = self.R_im[walk[1], :]
        r_re = self.R_re[walk[1], :]

        tmp3 = tmp1*r_re + tmp2*r_im
        tmp4 = tmp2*r_re - tmp1*r_im

        r_im = self.R_im[walk[2], :]
        r_re = self.R_re[walk[2], :]

        tmp5 = h_im*r_re + h_re*r_im
        tmp6 = h_re*r_re - h_im*r_im

        result = tmp3@tmp5 + tmp4@tmp6

        return result


        

