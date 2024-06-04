import pickle
import os
import torch
import numpy as np

class Reachability():
    def __init__(self, prefix):
        self.path = prefix
        self.reach_list, self.num_ent = self.get_reach()

    def get_reach(self):
        with open(self.path, 'r') as fi:
            for idx, line in enumerate(fi):
                line = [int(_) for _ in line.strip().split()]
                if (idx == 0):
                    num_ent = line[0]
                elif (idx == 1):
                    y = line
                elif (idx == 2):
                    indptr = line
        reach_list = [[] for _ in range(num_ent)]
        for _ in range(len(indptr) - 1):
            reach_list[_] = y[indptr[_]:indptr[_ + 1]]
        return reach_list, num_ent

    def get_factor(self, h):
        head = h[:,0].squeeze()
        # print(head)
        weights = np.array([[0.5 for _ in range(self.num_ent)] for _ in head.tolist()])
        # print(weights)
        indices = [self.reach_list[_] for _ in head.tolist()]
        for idx,_ in enumerate(indices):
            weights[idx,_] = 1
        return torch.IntTensor(weights)



class SimKGC():
    def __init__(self, dataset, prefix, nbf2rnnent, nbf2rnnrel):
        self.prefix = prefix
        self.h_embs = []
        self.re_index = [nbf2rnnent[_] for _ in range(len(nbf2rnnent))]
        self.re_rel = [nbf2rnnrel[_] for _ in range(len(nbf2rnnrel))]

        t_file = os.path.join(prefix, f'{DATASET}_Vectors/SimKGC_t_rep.pkl')
        t_emb = pickle.load(open(t_file, 'rb'))
        t_emb = t_emb[self.re_index, :]
        self.t_emb = t_emb.t().cpu()
        print(self.t_emb.shape)
        print(f'Loaded for tail')

        for _ in self.re_rel:
            print(f'Loading for relation {_}')
            h_file = os.path.join(prefix, f'{DATASET}_Vectors/SimKGC_h_{_}_rep.pkl')
            h_emb = pickle.load(open(h_file, 'rb'))
            h_emb = h_emb[self.re_index, :]
            self.h_embs.append(h_emb.cpu())
            print(f'Loaded for relation {_}')
        self.h_embs = torch.stack(self.h_embs).cpu()
        print(self.h_embs.shape)

    def get_h(self, h, r):
        # print(h,r)
        head = h[:,0].squeeze()
        rel = r[:,0].squeeze()
        h_req = self.h_embs[rel, head]
        t_req = self.t_emb
        # print(h_req.shape)
        # print(t_req.shape)
        result = torch.matmul(h_req, t_req)
        # print(result.shape)
        return result

    def get_rep(self, h, r, t=None):
        head = h[:,0].squeeze()
        rel = r[:,0].squeeze()
        h_req = self.h_embs[rel, head]
        # t_req = self.t_emb.t()[t]
        # print(h_req.shape)
        # print(h.shape)
        # print(t_req.shape)
        # result = torch.cat([h_req, t_req], dim=-1)
        # print(result.shape)
        return h_req

    
