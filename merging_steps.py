from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm

def merging_step1(candidate_list):
    """
    In the first merging step, we merge two candidates if the head of each of their representative phrase 
     is identical by string comparison.
    """

    indices_to_remove = set()
    for up_cand_id in tqdm(range(len(candidate_list))):   

        if up_cand_id in indices_to_remove:
            continue
        up_cand = candidate_list[up_cand_id]    
            
        for low_cand_id in range(up_cand_id+1,len(candidate_list)):
            low_cand = candidate_list[low_cand_id]

            if up_cand[1].lower() == low_cand[1].lower():# and up_cand[3] == low_cand[3]:


                indices_to_remove.add(low_cand_id)

                
    return indices_to_remove


def merge_indices(cand_df,indices_to_remove):                

    print(f'Initial amount of candidates: {len(cand_df)}')                
    #print(len(sorted(indices_to_remove)))

    #for index in reversed(sorted(indices_to_remove)):
    cand_df.drop(indices_to_remove,inplace=True)
        
    cand_df.reset_index(drop=True,inplace=True)
    print(f'Amount of candidates: {len(cand_df)}, after removing {len(sorted(indices_to_remove))} indices') 
    return cand_df



def merging_step2(candidate_list, model, what_merged2):
    
    indices_to_remove = set()
    
    for up_cand_id in tqdm(range(len(candidate_list))):     
        up_cand = candidate_list[up_cand_id]
        
        up_cand_mean_vec = phrase_heads_avg_vector(up_cand[2],model)
        
        for low_cand_id in range(up_cand_id+1,len(candidate_list)): 
            low_cand = candidate_list[low_cand_id]
            #print(f'for index {candidate_list.index(longer_cand)} checking the index {candidate_list.index(cand)}')
            #if candidate_list[longer_cand][1] == candidate_list[cand][1]:
                #print(f'matching "{longer_cand}" with "{cand}"')
            low_cand_mean_vec = phrase_heads_avg_vector(low_cand[2],model)
            similarity = 1-cosine(up_cand_mean_vec,low_cand_mean_vec)
            if up_cand[3] == low_cand[3]:
                if similarity >= 0.70:
                    print(f'matching "{up_cand[0]}" with "{low_cand[0]}" of the same type with {similarity} sim') 
                    indices_to_remove.add(low_cand_id)
                    what_merged2[up_cand[0].lower()].append(low_cand)
                        

            else:
                if similarity >= 0.8:
                    print(f'matching "{up_cand[0]}" with "{low_cand[0]}" of diff type with {similarity} sim') 
                    indices_to_remove.add(low_cand_id)
                    what_merged2[up_cand[0].lower()].append(low_cand)



    return indices_to_remove, what_merged2

def phrase_heads_avg_vector(phrase_set,model):
    phrase_head_vectors = []
    for phrase_head in phrase_set:    
        try:
            phrase_head_vectors.append(model.wv[phrase_head.lower()])
        except KeyError:
            #phrase_head_vectors.append(np.NaN)
            pass
    #phrase_head_vectors = [model[phrase_head] for phrase_head in phrase_set]
    if len(phrase_head_vectors) != 0:
        return np.mean(phrase_head_vectors,axis=0)
    else: 
        return np.NaN


from sklearn.cluster import AffinityPropagation

from sklearn.metrics.pairwise import cosine_similarity

def merging_step3(cand_df,model,what_merged3):
    phrases = []
    indices_to_remove = set()
    # 1. first we find adj-nn phrases within the candidate
    for candidate in cand_df['cand_tags']:  
        #the head of noun phrase is marked with value 0 for the word.head
        cand_heads_pos = [(word.text, word.head, word.xpos) for word in candidate.words]
        #np_pos_tags = {word.text: word.xpos for sent in doc.sentences for word in sent.words}
        #print(np_heads_pos)
        cand_labeling_phrases = []
        for word, head, pos in cand_heads_pos:
            #head-1 because the pointer to head does not use 0 index
            if (pos == 'JJ' or pos=='VBN') and 'NN' in cand_heads_pos[head-1][2]:
                cand_labeling_phrases.append(f'{word}_{cand_heads_pos[head-1][0]}')
        phrases.append(cand_labeling_phrases)
    
    candidate_list = cand_df['candidates']
    # 2. we compare the similarities of candidates' phrases
    for up_cand_id in tqdm(range(len(candidate_list))):     
        up_cand = candidate_list[up_cand_id]
        up_cand_vectors = phrases_vectors(phrases[up_cand_id],model)
        if len(up_cand_vectors)==0:
            pass
        else:
            for low_cand_id in range(up_cand_id+1,len(candidate_list)): 
                low_cand = candidate_list[low_cand_id]
                low_cand_vectors = phrases_vectors(phrases[low_cand_id],model)
                if len(low_cand_vectors)==0:
                    pass
                else:
                    sim_matrix = np.zeros((len(up_cand_vectors),len(low_cand_vectors)))
                    #print(sim_matrix)
                    for i in range(len(up_cand_vectors)):
                        for j in range(len(low_cand_vectors)):

                            sim_matrix[i][j] = 1-cosine(up_cand_vectors[i],low_cand_vectors[j])

                    # can we compute matrix mean like this? 
                    #print(sim_matrix)
                    if np.mean(sim_matrix) > 0.6:
                        #print(f'{longer_cand} and {cand} are {numpy.mean(sim_matrix)} similar' )
                        indices_to_remove.add(low_cand_id)
                        what_merged3[up_cand[0].lower()].append(low_cand)
                    #else:
                        #print(f'{numpy.mean(sim_matrix)} is not similar' )
                    
    return indices_to_remove, what_merged3
                


def phrases_vectors(cand_phrases,model):
    
#for cand_phrases in phrases:
    #print(cand_phrases)
    cand_phrase_vectors = []
    for phrase in cand_phrases:
        try:
            cand_phrase_vectors.append(model.wv[phrase.lower()])
            #print(f'for existing phrase "{phrase}" the vector is {model[phrase][0]}')
        except KeyError:
            phrase_words = phrase.split('_')
            #print(model[phrase_words[1]])
            try:
                phrase_vectors = [model.wv[phrase_word.lower()] for phrase_word in phrase_words]
                #print(f'for phrase "{phrase}" avg vector is "{sum(phrase_vectors)/len(phrase_vectors)}') 
                cand_phrase_vectors.append(np.nanmean(phrase_vectors))
            except KeyError:
                cand_phrase_vectors.append(np.NaN)
    #print(len(cand_phrase_vectors))
    return cand_phrase_vectors
    


# missing the second method - we check for the lexical identity of specific stems in multiple candidates.

def merging_step4(cand_df,model,what_merged4):
    phrases = []
    indices_to_remove = set()
    # 1. first we find adj-nn phrases within the candidate
    for candidate in cand_df['cand_tags']:

        #the head of noun phrase is marked with value 0 for the word.head
        cand_heads_pos = [(word.text, word.head, word.xpos) for word in candidate.words]

        #print(np_heads_pos)
        cand_compound_phrases = []
        for word, head, pos in cand_heads_pos:
            #i = np_heads_pos.index((word, head, pos))
            #print(np_heads_pos)
            #print(np_heads_pos[i])
            #print(np_heads_pos[head-1])
            #'NN' in np_heads_pos[head-1][2] and
            try:
                #if 'NN' in pos and 'NN' in cand_heads_pos[i+1][2] : 
                    #cand_compound_phrases.append(f'{word}_{cand_heads_pos[i+1][0]}')
                if 'NN' in pos and 'NN' in cand_heads_pos[head-1][2]:
                    cand_compound_phrases.append(f'{word}_{cand_heads_pos[head-1][0]}')
            except IndexError:
                pass
        phrases.append(cand_compound_phrases)
    
    candidate_list = cand_df['candidates']
    # 2. we compare the similarities of candidates' phrases
    for up_cand_id in tqdm(range(len(candidate_list))):     
        up_cand = candidate_list[up_cand_id]
        up_cand_vectors = phrases_vectors(phrases[up_cand_id],model)
        if len(up_cand_vectors)==0:
            pass
        else:
            for low_cand_id in range(up_cand_id+1,len(candidate_list)):
                low_cand = candidate_list[low_cand_id]
                low_cand_vectors = phrases_vectors(phrases[low_cand_id],model)
                if len(low_cand_vectors)==0:
                    pass
                else:
                    sim_matrix = np.zeros((len(up_cand_vectors),len(low_cand_vectors)))
                    #print(sim_matrix)
                    for i in range(len(up_cand_vectors)):
                        for j in range(len(low_cand_vectors)):
                            #print(cosine_similarity(long_cand_vectors[i].reshape(1,-1),short_cand_vectors[j].reshape(1,-1)))
                            sim_matrix[i][j] = 1-cosine(up_cand_vectors[i],low_cand_vectors[j])
                            """if cosine_similarity(long_cand_vectors[i].reshape(1,-1),short_cand_vectors[j].reshape(1,-1)) > 0.4:                
                                sim_matrix[i][j] = 2
                            elif cosine_similarity(long_cand_vectors[i].reshape(1,-1),short_cand_vectors[j].reshape(1,-1)) > 0.2:
                                sim_matrix[i][j] = 1
                            else:
                                sim_matrix[i][j] = 0"""

                    #print(sim_matrix, up_cand,low_cand)            
                    if np.mean(sim_matrix) > 0.6:
                        #print(f'{up_cand_id} and {low_cand_id} are {np.mean(sim_matrix)} similar' )
                        indices_to_remove.add(low_cand_id)
                        what_merged4[up_cand[0].lower()].append(low_cand)
                    #else:
                        #print(f'{numpy.mean(sim_matrix)} is not similar' )
                    
    return indices_to_remove, what_merged4


def merge_transitively(what_merged):
    keys_to_delete = set()
    for key,values in what_merged.items():   
        for value in values:
            #print(what_merged.keys())
            if value[0] in what_merged.keys():
                #print(what_merged[key])
                what_merged[key].append(value)
                what_merged[key] = list(what_merged[key])
                keys_to_delete.add(value[0])

    for key in list(keys_to_delete):
        what_merged.pop(key)  
    return what_merged