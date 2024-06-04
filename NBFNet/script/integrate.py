import json

nbfrel = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }

def get_WN_mappings(dataset, root, rotate = False):
    if (dataset == 'WN18RR'):
        PREFIX = root +'/raw/'
        NUM_RELATIONS = 11
    elif (dataset == 'FB15k-237'):
        PREFIX = root + '/FB15k-237/raw/'
        NUM_RELATIONS = 237
    elif (dataset == 'CodexM'):
        PREFIX = root
        NUM_RELATIONS = 71
    else:
        PREFIX = root
        NUM_RELATIONS = 37
    rnnrel = {}
    with open(PREFIX + 'relations.dict', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            rnnrel[line[1]] = int(line[0])
    rnnent = {}
    with open(PREFIX + 'entities.dict', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            rnnent[line[1]] = int(line[0])
    nbfent = {}
    idx = 0
    src = []
    dst = []
    with open(PREFIX + 'train.txt', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            src.append(line[0])
            dst.append(line[2])
    for _ in src:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    for _ in dst:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    src = []
    dst = []
    with open(PREFIX + 'valid.txt', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            src.append(line[0])
            dst.append(line[2])
    for _ in src:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    for _ in dst:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    src = []
    dst = []
    with open(PREFIX + 'test.txt', 'r') as fi:
        for line in fi:
            line = line.strip().split()
            src.append(line[0])
            dst.append(line[2])
    for _ in src:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    for _ in dst:
        if _ not in nbfent:
            nbfent[_] = idx
            idx += 1
    if (WN == 0):
        nbf2rnnent = {}
        for _ in nbfent:
            nbf2rnnent[nbfent[_]] = rnnent[_]
        nbf2rnnrel = {}
        for _ in nbfrel:
            nbf2rnnrel[nbfrel[_]] = rnnrel[_]
            nbf2rnnrel[NUM_RELATIONS + nbfrel[_]] = NUM_RELATIONS + rnnrel[_]
        return nbf2rnnent, nbf2rnnrel, nbfent, nbfrel
    elif (WN == 1):
        # simrel = {}
        # with open(PREFIX + 'relations.json', 'r') as fi:
        #     for line in fi:
        #         line = line.strip()
        #         if (len(line) <= 1):
        #             continue
        #         line = line.split(':')
        #         simrel[line[0].strip()[1:-1]] = len(simrel)
        if (rotate):
            nbf2rnnent = {_:_ for _ in range(len(rnnent))}
        else:
            ent = json.load(open(PREFIX + 'entities.json', 'r'))
            siment = [_["entity_id"] for _ in ent]
            siment = {siment[_]:_ for _ in range(len(siment))}
            nbf2rnnent = {}
            for _ in rnnent:
                nbf2rnnent[rnnent[_]] = siment[_]
        # nbf2rnnrel = {}
        # for _ in rnnrel:
        #     nbf2rnnrel[rnnrel[_]] = simrel[_]
        #     nbf2rnnrel[NUM_RELATIONS + rnnrel[_]] = NUM_RELATIONS + simrel[_]
        nbf2rnnrel = {_:_ for _ in range(2*NUM_RELATIONS)}
        return nbf2rnnent, nbf2rnnrel, rnnent, rnnrel 
    elif (WN==2):
        nbf2rnnent = {_:_ for _ in range(len(rnnent))}
        nbf2rnnrel = {_:_ for _ in range(2*NUM_RELATIONS)}
        return nbf2rnnent, nbf2rnnrel, rnnent, rnnrel 
    else:
        if (rotate):
            nbf2siment = {_:_ for _ in range(len(rnnent))}
        else:
            nbf2siment = {}
            entities = json.load(open(PREFIX + 'entities.json', 'r'))
            for _ in entities:
                nbf2siment[_["entity_id"]] = len(nbf2siment)
        nbf2simrel = {_:_ for _ in range(2*NUM_RELATIONS)}
        return nbf2siment, nbf2simrel, rnnent, rnnrel


# def get_index_dict(WN=True):
#     index_dict = {}
#     if (WN):
#         nbf2rnnent, nbf2rnnrel = get_WN_mappings()
#         rnn2nbfent = {nbf2rnnent[_]:_ for _ in nbf2rnnent}
#         rnn2nbfrel = {nbf2rnnrel[_]:_ for _ in nbf2rnnrel}
#         raw = json.load(open(PREFIX+'SimKGC_scores_for_SimKGC_WN18RR_no_desc.json', 'r'))
#         for _ in raw:
#             head = rnn2nbfent[int(_["h"])]
#             rel = int(_["r"])
#             if (rel >= 11):
#                 rel = 11 + rnn2nbfrel[rel - 11]
#             else:
#                 rel = rnn2nbfrel[rel]
#             index = [rnn2nbfent[i] for i in _["index"]]
#             index_dict[(head, rel)] = index
#     else:
#         raw = json.load(open(PREFIX+'SimKGC_scores_for_SimKGC_FB15k237_no_desc.json', 'r'))
#         for _ in raw:
#             head = int(_["h"])
#             rel = int(_["r"])
#             index = _["index"]
#             index_dict[(head, rel)] = index
#     return index_dict

# def get_SimKGC_scores(WN=True):
#     index_dict = {}
#     if (WN):
#         nbf2rnnent, nbf2rnnrel = get_WN_mappings()
#         rnn2nbfent = {nbf2rnnent[_]:_ for _ in nbf2rnnent}
#         rnn2nbfrel = {nbf2rnnrel[_]:_ for _ in nbf2rnnrel}
#         raw = json.load(open(PREFIX+'SimKGC_scores_for_NBF_WN18RR.json', 'r'))
#         for _ in raw:
#             head = rnn2nbfent[int(_["h"])]
#             rel = int(_["r"]) 
#             if (rel >= 11):
#                 rel = 11 + rnn2nbfrel[rel - 11]
#             else:
#                 rel = rnn2nbfrel[rel]
#             index = [rnn2nbfent[i] for i in _["index"]]
#             scores = _["score"]
#             index2score = {}
#             for i in range(len(index)):
#                 index2score[index[i]] = scores[i]
#             index_dict[(head, rel)] = index2score
#     else:
#         raw = json.load(open(PREFIX+'SimKGC_scores_for_NBF_FB15k237.json', 'r'))
#         for _ in raw:
#             head = int(_["h"])
#             rel = int(_["r"])
#             index = _["index"]
#             scores = _["score"]
#             index2score = {}
#             for i in range(len(index)):
#                 index2score[index[i]] = scores[i]
#             index_dict[(head, rel)] = index2score
#     return index_dict

