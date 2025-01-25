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
        # print("CUR: ", cur)
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

def sym_dif(node1: str, node2: str):
    set1 = set(node1.split())
    set2 = set(node2.split())
    dif = list(set1.symmetric_difference(set2))
    first = re.sub(r'[^\w\s]','',dif[0])
    second = re.sub(r'[^\w\s]','',dif[1])
    score = max(evaluator.score([(first,second)]), evaluator.score([(second,first)]))
    one_word = len(dif) == 2 and score < env_settings.ONE_WORD_TH
    return one_word

def ends_match(node1: str, node2: str):
    end1 = node1.rstrip()
    end2 = node2.rstrip()
    doc1 = nlp(node1)
    doc2 = nlp(node2)
    sents1 = [str(sent) for sent in doc1.sents]
    sents2 = [str(sent) for sent in doc2.sents]
    return re.match("^.*(?<![!?])$",node1) and re.match("^.*(?<![!?])$",node2) or all([end1,end2,end1[-1] == end2[-1]]), sents1, sents2

def if_greetings(node1: str, node2:str):
    greetings = re.compile(r'\b(welcome|good morning|good evening|good afternoon|hi|hello|hey|thank|thanks|great|awesome|perfect|fantastic|wonderful)\b', re.IGNORECASE)
    if (re.search(greetings, node1) is None) is not (re.search(greetings, node2) is None):
        greetings_cond = True
        print("GREETINGS")
    else:
        greetings_cond = False
    return greetings_cond

def if_else(node1: str, node2:str):
    else_re = re.compile(r'\b(else|add|another|more|other|extra|additional)\b', re.IGNORECASE)
    if (re.search(else_re, node1) is None) is not (re.search(else_re, node2) is None):
        else_cond = True
        print("ELSE")
    else:
       else_cond = False
    return else_cond

def y_n(node1: str, node2:str):
    yn = re.compile(r'\b(where|what|when|where|who|whom|which|whose|why|how)\b', re.IGNORECASE)
    if (re.search(yn, node1) is None) is not (re.search(yn, node2) is None):
        yes_no = True
        print("Y_N")
    else:
        yes_no = False
    return yes_no

def if_if(node1: str, node2:str):
    if_re = re.compile(r'\b(if|whether)\b', re.IGNORECASE)
    if (re.search(if_re, node1) is None) is not (re.search(if_re, node2) is None):
        if_cond = True
        print("IF")
    else:
        if_cond = False
    return if_cond

def nodes2groups(nodes_list: list[str], next_list: list[str], mix_list: list[str], neigbhours: dict):
    """ Rule based algorithm to group graph nodes
    nodes_list: list of assistant's utterances
    next_list: list of user's utterances
    mix_list: list of nodes and edges concatenation
    neighbours: dictionary of adjacent nodes
    Based on cross-encoder similarity and some more empirical rules
    """

    nodes_score = get_reranking(nodes_list, nodes_list)
    next_score = get_reranking(next_list, next_list)
    mix_score = get_reranking(mix_list, mix_list)
    pairs = []

    for ind1, node1 in enumerate(nodes_list):
        cur_nodes_list = nodes_list[ind1+1:]
        for ind2, node2 in zip(range(ind1+1,ind1+1+len(cur_nodes_list)),cur_nodes_list):

            max_n = max(nodes_score[ind1][ind2],nodes_score[ind2][ind1])
            max_m = max(mix_score[ind1][ind2],mix_score[ind2][ind1])
            min_e = min(next_score[ind1][ind2],next_score[ind2][ind1])
            max_e = max(next_score[ind1][ind2],next_score[ind2][ind1])

            print(f"MIX: max_n:{max_n},max_m:{max_m},min_e:{min_e},max_e:{max_e}",nodes_list[ind1],nodes_list[ind2])
            one_word = sym_dif(node1, node2)
            signs, sents1, sents2 = ends_match(node1, node2)
            if not signs:
                print("SIGNS")
            len1 = len(sents1)
            len2 = len(sents2)
            greetings_cond = if_greetings(node1, node2)
            if_cond = if_if(node1, node2)
            else_cond = if_else(node1, node2)
            yes_no = y_n(node1, node2)

            condition = not (greetings_cond or if_cond or else_cond or yes_no or one_word or (len1 == 1 and len2 == 1 and not signs) or node1 in neigbhours[node2])
            cond_1 = len1 == len2 and len1 == 1 and min_e >= 0.06 and max_e > 0.9 and max_m >= env_settings.NEXT_RERANKER_THRESHOLD and max_n > 0.05 or len1!=len2 and max_n > 0.99
            if cond_1:
                condition = not (greetings_cond or if_cond or else_cond or one_word or (len1 == 1 and len2 == 1 and not signs) or node1 in neigbhours[node2])

            if condition:
                if cond_1:
                    print("FIRST: ", node1, node2)
                    pairs.append(((max_n+max_m)/2,(ind1,ind2)))
                else:

                    sent_score = []
                    if len1 > 1 and len1 == len2:


                        for s1,s2 in zip(sents1,sents2):
                            sent_score.append((s1,s2))
                            sent_score.append((s2,s1))
                        sent_score = evaluator.score(sent_score)
                        maxes = [max(el[0],el[1]) for el in zip(sent_score[::2],sent_score[1::2])]
                        nodes_condition = max(maxes) >= 0.9 and min(maxes) >= 0.05 and max_n >= env_settings.RERANKER_THRESHOLD
                    else:
                        nodes_condition = len1==len2 and max_n >= env_settings.RERANKER_THRESHOLD 

                    if max_m >= env_settings.NEXT_RERANKER_THRESHOLD and nodes_condition:
                        print("SECOND: ", node1, node2)
                        pairs.append(((max_n+max_m)/2,(ind1,ind2)))
    groups = unite_pairs(pairs)
    grouped = []
    for el in groups:
        grouped += el
    singles = [[idx] for idx in range(len(nodes_list)) if idx not in grouped]
    groups += singles
    print("INDEX: ", groups)
    groups = [[nodes_list[el] for el in g] for g in groups]

    return groups

# def nodes2clusters(nodes_list: list[str]):
#     # nodes_scores = get_reranking(nodes_list,nodes_list)
#     # mix_scores = get_reranking(mix_list,mix_list)

#     nodes_scores = get_embedding(nodes_list,nodes_list, "BAAI/bge-m3", 'cpu')
#     # mix_scores = get_embedding(mix_list,mix_list, "BAAI/bge-m3", 'cpu')


#     return DBSCAN(min_samples=1).fit_predict(nodes_scores)

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
