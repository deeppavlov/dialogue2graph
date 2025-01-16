import copy
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import spacy 

from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator

from chatsky_llm_autoconfig.settings import EnvSettings
env_settings = EnvSettings()
embedding = {}


nlp = spacy.load('en_core_web_sm')
evaluator = HuggingFaceCrossEncoder(model_name=env_settings.RERANKER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})

# def compare_strings(first: str, second: str):

#     # score = evaluator.evaluate_string_pairs(prediction=first, prediction_b=second)['score']
#     score = evaluator.score([(first, second)])[0]
#     # if 0.99 > score > 0.94:
#     #     print("SCORE: ", score, first, second)
#     # return score <= env_settings.EMBEDDER_THRESHOLD
#     return score >= env_settings.RERANKER_THRESHOLD

def compare_strings(first: str, second: str, embeddings: HuggingFaceEmbeddings):

    evaluator_2 = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
    score = evaluator_2.evaluate_string_pairs(prediction=first, prediction_b=second)['score']
    return score <= env_settings.EMBEDDER_THRESHOLD


class EmbeddableString:
    def __init__(self, element: str):
        self.element = element
    def __eq__(self, other):
        return compare_strings(self.element,other.element)
    def __hash__(self):
        return hash("")
    def __str__(self):
        return self.element

def emb_list(x):
    #print("EMB_LIST: ", x)
    return [EmbeddableString(el) for el in x]

def get_embedding(generated: list[str], golden: list[str], emb_name: str, device: str):

    if emb_name not in embedding:
        embedding[emb_name] = SentenceTransformer(emb_name,device=device)
 
    golden_vectors = embedding[emb_name].encode(golden, normalize_embeddings=True)
    generated_vectors = embedding[emb_name].encode(generated, normalize_embeddings=True)
    similarities = generated_vectors @ golden_vectors.T
    return similarities

def get_reranking(generated: list[str], golden: list[str]):
    
    sz = len(generated)
    to_score = []
    for gen in generated:
        for gol in golden:
            to_score.append((gen,gol))
    print("SCORING...")
    # print(to_score)
    score = np.array(evaluator.score(to_score))
    print("finished")

    return score.reshape(sz,sz)

def get_cross(search, list2):

    sublist = [p for p in list2 if search in p]
    to_add = []
    for s in sublist:
        if s[0]!=search:
            to_add.append(s[0])
        else:
            to_add.append(s[1])
    return len(to_add)

def unite_pairs(pairs: list[tuple[float,tuple[int,int]]]):
    pairs_in = copy.deepcopy(pairs) 
    pairs_in.sort(reverse=True)
    pairs_in = [x[1] for x in pairs_in]
    groups = []
    while pairs_in:
        cur = [p for p in pairs_in if p[0] in p or p[1] in p]
        print("CUR: ", cur)
        # for x in cur:
        x = cur[0]
        list1 = [p for p in cur if x[0] in p and x!=p]
        list2 = [p for p in cur if x[1] in p and x!=p]
        print("LIST1: ", list1)
        print("LIST2: ", list2)
        to_add = []
        for y in list1:
            to_add = []
            for el in x:
                if el == y[0]:
                    search = y[1]
                else:
                    search= y[0]
                if get_cross(search,list2):
                    to_add += [search]
            print("TOADD: ", to_add)

            # if x[0] == y[0]:
            #     search = y[1]
            # else:
            #     search= y[0]
            # to_add = []
            # if get_cross(search,list2):
            #     to_add = [search]
            # print("TOADD: ", to_add)
    
            # if x[1] == y[0]:
            #     search = y[1]
            # else:
            #     search= y[0]

            # if get_cross(search,list2):
            #     to_add += [search]

            # print("TO ADD: ", to_add)
                # to_add += list(set(to_add))
        # Дальше надо объединить их и удалить, потом удаление
        to_add = list(set(([x[0],x[1]]+to_add)))
        groups.append(to_add)
        pairs_in = [p for p in pairs_in if p[0] not in to_add and p[1] not in to_add]

        print("TO_ADD: ", to_add)
        print("LEFT: ", pairs_in)
    return groups

def nodes2groups(nodes_list: list[str], next_list: list[str], mix_list: list[str]):
    
    sz = len(nodes_list)
    nodes_to_score = []
    nodes_to_back_score = []   
    mix_to_score = []
    mix_to_back_score = []
    next_to_score = []
    next_to_back_score = []
    index = []
    orig_index = []
    for ind1, en in enumerate(zip(mix_list, nodes_list)):
        mix1 = en[0]
        node1 = en[1]
        cur_mix_list = mix_list[ind1+1:]
        cur_nodes_list = nodes_list[ind1+1:]
        for ind2, mix2, node2 in zip(range(ind1+1,ind1+1+len(cur_mix_list)), cur_mix_list, cur_nodes_list):
            print("NNODES: ", node1, node2, ind1, ind2)
            set1 = set(node1.split())
            set2 = set(node2.split())
            dif = list(set1.symmetric_difference(set2))
            first = re.sub(r'[^\w\s]','',dif[0])
            second = re.sub(r'[^\w\s]','',dif[1])
            dif_score = min(evaluator.score([(first,second)]), evaluator.score([(second,first)]))
            print("DIF: ", dif, dif_score)
            end1 = node1.rstrip()
            end2 = node2.rstrip()
            doc1 = nlp(node1)
            doc2 = nlp(node2)
            sents1 = [str(sent) for sent in doc1.sents]
            sents2 = [str(sent) for sent in doc2.sents]
            signs = re.match("^.*(?<![!?])$",node1) and re.match("^.*(?<![!?])$",node2) or all([end1,end2,end1[-1] == end2[-1]])

            orig_index.append((ind1, ind2))
            one_word = len(dif) == 2 and dif_score < env_settings.RERANKER_THRESHOLD
            if one_word:
                print("ONE_WORD: ", node1, node2)
            if ind1 < sz-1 and nodes_list[ind1+1] == node2:
                print("Neighbors: ", node1, node2)
            if not signs:
                print("Signs: ", node1, node2)                
            if one_word or len(sents1) != len(sents2) or not signs \
                or ind1 < sz-1 and nodes_list[ind1+1] == node2: # adjacent utterances cannot be one node:
                
                nodes_to_score.append(("no","yes"))
                nodes_to_back_score.append(("no","yes"))

            else:




                to_score = [(node1,node2)]
                to_back_score = [(node2,node1)]
                
                if len(sents1) > 1:
                    for s1,s2 in zip(sents1,sents2):
                        to_score.append((s1,s2))
                        to_back_score.append((s2,s1))
                        orig_index.append((ind1, ind2))

                nodes_to_score.extend(to_score)
                nodes_to_back_score.extend(to_back_score)


            mix_to_back_score.append((mix2,mix1))
            mix_to_score.append((mix1,mix2))
            next_to_back_score.append((mix2,mix1))
            next_to_score.append((mix1,mix2))
            index.append((ind1, ind2))

    print("SCORING...")
    # print(to_score)
    nodes_score = evaluator.score(nodes_to_score)
    mix_score = evaluator.score(mix_to_score)
    nodes_back_score = evaluator.score(nodes_to_back_score)
    mix_back_score = evaluator.score(mix_to_back_score)
    next_score = evaluator.score(next_to_score)
    next_back_score = evaluator.score(next_to_back_score)

    #     n_0 = n[0].rstrip()
    #     n_1 = n[1].rstrip()
    #     doc1 = nlp(n[0])
    #     doc2 = nlp(n[1])
    #     tokens1 = [[token.text for token in sent] for sent in doc1.sents]
    #     tokens2 = [[token.text for token in sent] for sent in doc2.sents]
    #     signs = n_0 and n_1 and ((n_0[-1] == '!' and n_1[-1] != '!') or (n_0[-1] == '?' and n_1[-1] != '?') or (n_1[-1] == '!' and n_0[-1] != '!') or (n_1[-1] == '?' and n_0[-1] != '?'))
    #     if signs or len(tokens1)!=len(tokens2):
    #         nodes_score[idx] = 0
    #         nodes_back_score[idx] = 0


#  >= env_settings.RERANKER_THRESHOLD
    # print("SCORE: ", score)
    print("finished")

    pairs = []
    orig_idx = 0
    for idx, mix in enumerate(zip(mix_score, mix_back_score, next_score, next_back_score)):
        sz = len([o for o in orig_index if index[idx] == o])

        max_n = max(nodes_score[orig_idx], nodes_back_score[orig_idx])
        maxes = [max(el[0],el[1]) for el in zip(nodes_score[orig_idx:orig_idx+sz],nodes_back_score[orig_idx:orig_idx+sz])]
        if sz > 1:
            nodes_condition = max(maxes[1:]) >= 0.9 and min(maxes[1:]) >= 0.05 and maxes[0] >= env_settings.RERANKER_THRESHOLD
        else:
            nodes_condition = maxes[0] >= env_settings.RERANKER_THRESHOLD
        orig_idx += sz

        # print("MIX: ",max(en[0],en[1]),max(en[2],en[3]),nodes_to_score[idx])
        max_m = max(mix[0],mix[1])
        min_e = min(mix[2],mix[3])
        if min_e < 0.06:
            condition = nodes_condition
        else:
            condition = max_m >= env_settings.NEXT_RERANKER_THRESHOLD and nodes_condition
        if condition:
            pairs.append(((max_n+max_m)/2,index[idx]))
        print("SCORES: ",max_m,max_n,min_e, nodes_to_score[idx])

    # for idx, en in enumerate(zip(mix_score, mix_back_score, nodes_score, nodes_back_score)):
    #     print("MIX: ",max(en[0],en[1]),max(en[2],en[3]),nodes_to_score[idx])
    #     max1 = max(en[0],en[1])
    #     max2 = max(en[2],en[3])
    #     if max1 >= env_settings.NEXT_RERANKER_THRESHOLD and max2 >= env_settings.RERANKER_THRESHOLD:
    #         pairs.append(((max1+max2)/2,index[idx]))

    print("PAIRS: ", pairs)

    groups = unite_pairs(pairs)
    grouped = []
    for el in groups:
        grouped += el
    singles = [[idx] for idx in range(len(nodes_list)) if idx not in grouped]
    groups += singles
    print("INDEX: ", groups)
    groups = [[nodes_list[el] for el in g] for g in groups]

    # groups = []
    # firsts = [p[0] for p in pairs]
    # for idx, g in enumerate(nodes_list):
    #     if idx in firsts:
    #         flag = 0
    #         node = nodes_list[pairs[firsts.index(idx)][1]]
    #         for idx_2, gr in enumerate(groups):
    #             if any([g==el for el in gr]):
    #                 flag = 1
    #                 if node not in groups[idx_2]:
    #                     groups[idx_2].append(node)
    #                     print("ANY adding  ",node,idx_2)
    #                 break
    #         if not flag:
    #             groups.append([node,g])
    #             print("flag adding:  ",node,g)
    #     else:
    #         if all([g!=el for group in groups for el in group]):
    #             groups.append([g])
    #             print("else adding:  ",g)
    return groups

from sklearn.cluster import DBSCAN

# def nodes2clusters(nodes_list: list[str], mix_list: list[str]):
#     # nodes_scores = get_reranking(nodes_list,nodes_list)
#     # mix_scores = get_reranking(mix_list,mix_list)

#     nodes_scores = get_embedding(nodes_list,nodes_list, "BAAI/bge-m3", 'cpu')
#     mix_scores = get_embedding(mix_list,mix_list, "BAAI/bge-m3", 'cpu')

#     scores = np.mean( np.array([ nodes_scores, mix_scores ]), axis=0 )

#     return DBSCAN(min_samples=1).fit_predict(scores)

def nodes2clusters(nodes_list: list[str]):
    # nodes_scores = get_reranking(nodes_list,nodes_list)
    # mix_scores = get_reranking(mix_list,mix_list)

    nodes_scores = get_embedding(nodes_list,nodes_list, "BAAI/bge-m3", 'cpu')
    # mix_scores = get_embedding(mix_list,mix_list, "BAAI/bge-m3", 'cpu')


    return DBSCAN(min_samples=1).fit_predict(nodes_scores)


# def nodes2groups(nodes_list: list[str], mix_list: list[str]):
    
#     sz = len(nodes_list)
#     nodes_to_score = []  
#     mix_to_score = []
#     nodes_scores = get_reranking(nodes_list,nodes_list)
#     mix_scores = get_reranking(mix_list,mix_list)
#     index = []
#     for ind1, en in enumerate(zip(mix_list, nodes_list)):
#         mix1 = en[0]
#         node1 = en[1]
#         cur_mix_list = mix_list[ind1+1:]
#         cur_nodes_list = nodes_list[ind1+1:]
#         for ind2, mix2, node2 in zip(range(ind1+1,ind1+1+len(cur_mix_list)), cur_mix_list, cur_nodes_list):
#             mix_to_score.append((mix1,mix2))
#             nodes_to_score.append((node1,node2))
#             mix_to_back_score.append((mix2,mix1))
#             nodes_to_back_score.append((node2,node1))
#             index.append((ind1, ind2))
#     print("SCORING...")
#     # print(to_score)
#     nodes_score = evaluator.score(nodes_to_score)
#     mix_score = evaluator.score(mix_to_score)
#     nodes_back_score = evaluator.score(nodes_to_back_score)
#     mix_back_score = evaluator.score(mix_to_back_score)
#     for idx, n in enumerate(nodes_to_score):
#         n_0 = n[0].rstrip()
#         n_1 = n[1].rstrip()
#         doc1 = nlp(n[0])
#         doc2 = nlp(n[1])
#         tokens1 = [[token.text for token in sent] for sent in doc1.sents]
#         tokens2 = [[token.text for token in sent] for sent in doc2.sents]
#         signs = n_0 and n_1 and ((n_0[-1] == '!' and n_1[-1] != '!') or (n_0[-1] == '?' and n_1[-1] != '?') or (n_1[-1] == '!' and n_0[-1] != '!') or (n_1[-1] == '?' and n_0[-1] != '?'))
#         if signs or len(tokens1)!=len(tokens2):
#             nodes_score[idx] = 0
#             nodes_back_score[idx] = 0



#     # print("SCORE: ", score)
#     print("finished")

#     pairs = []
#     for idx, en in enumerate(zip(mix_score, mix_back_score, nodes_score, nodes_back_score)):
#         print("MIX: ",max(en[0],en[1]),max(en[2],en[3]),nodes_to_score[idx])
#         if max(en[0],en[1]) >= env_settings.NEXT_RERANKER_THRESHOLD and max(en[2],en[3]) >= env_settings.RERANKER_THRESHOLD:
#             pairs.append(index[idx])

#     print("PAIRS: ", pairs)

#     groups = []
#     firsts = [p[0] for p in pairs]
#     for idx, g in enumerate(nodes_list):
#         if idx in firsts:
#             flag = 0
#             node = nodes_list[pairs[firsts.index(idx)][1]]
#             for idx_2, gr in enumerate(groups):
#                 if any([g==el for el in gr]):
#                     flag = 1
#                     if node not in groups[idx_2]:
#                         groups[idx_2].append(node)
#                         print("ANY adding  ",node,idx_2)
#                     break
#             if not flag:
#                 groups.append([node,g])
#                 print("flag adding:  ",node,g)
#         else:
#             if all([g!=el for group in groups for el in group]):
#                 groups.append([g])
#                 print("else adding:  ",g)
#     return groups




def get_2_rerankings(generated1: list[str], golden1: list[str], generated2: list[str], golden2: list[str]):
    
    sz = len(generated1)
    to_score = []
    for gen in generated1:
        for gol in golden1:
            to_score.append((gen,gol))
    for gen in generated2:
        for gol in golden2:
            to_score.append((gen,gol))
    print("SCORING...")
    # print(to_score)
    scores = np.array(evaluator.score(to_score))
    print("finished")

    return scores[:sz*sz].reshape(sz,sz), scores[sz*sz:].reshape(sz,sz)
