import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def get_tweet_tags(tagged_tweets):
    """
    Input: corpus of tweets to tag
    Output: list of tuples containing (POS-tags of each word, NER-tags of each named entity)
    """
    tweet_tags=[]
    for tweet in tqdm(range(len(tagged_tweets))):
            #extract POS and NE tags
            tweet_pos_tags={word.text: word.xpos for sent in tagged_tweets[tweet].sentences for word in sent.words}
            tweet_ner= {ent.text: ent.type for sent in tagged_tweets[tweet].sentences for ent in sent.ents}
            tweet_tags.append((tweet_pos_tags,tweet_ner))
    return tweet_tags  



# get noun phrases with tregex
def get_noun_phrases(client, tweet, annotators=None):
    """
    Input: client = CoreNLPClient instance
           tweet = tweet text
           annotators = allowed CoreNLP operations
    Output: list of all noun phrases in the tweet
    """
    pattern = 'NP'
    matches = client.tregex(tweet,pattern,annotators=annotators)

    return [sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]


def get_coref_chain(tweet,client):

    ann = client.annotate(tweet)        
    tweet_chains = ann.corefChain
    all_chains = list()
    
    
    for chain in tweet_chains:
        mychain = list()
        # Loop through every mention of this chain
        for mention in chain.mention:
            # Get the sentence in which this mention is located, and get the words which are part of this mention
            words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
            #build a string out of the words of this mention
            ment_word = ' '.join([x.word for x in words_list])
            
            mychain.append(ment_word)
            
        #the corefering words will be stored alongside the index of their representative in a tuple
        coref_group = (mychain,chain.representative)
        all_chains.append(coref_group)
    return all_chains

def prep_candlist_for_batching(cand_list):
    #change noun_phrase_list format to be batching compatible
    all_cands_list = cand_list.copy()
    for tweet_id in range(len(cand_list)):
        if len(cand_list[tweet_id]) == 0:
            cand_list[tweet_id] = ['candidate_to_be_removed']
     
        all_cands_list[tweet_id] = '\n\n'.join(cand_list[tweet_id])
    return all_cands_list



def get_cand_heads(tagged_cands):
    # each candidate will be stored as [(set_of_phrases_heads), cand_rep_head] 
    return [[set([cand.words[word.head-1].text for word in cand.words]), 
             [word.text for word in cand.words if word.head == 0]] #the root of NP has value 0. Since head is only one and stored in a list, we pick item [0]
             for cand in tagged_cands.sentences]# for tweet_cands in tagged_cands]



def get_synt_category(head):
    """
    Input: head word of the noun phrase e.g. 'aliens' from NP 'Illegal aliens' 
    Output: syntactic category of the head word as categorized using worndet
    """
    
    person_ss = wn.synsets("person")[0]
    #group_ss = wn.synsets("facility")[0]    
    place_ss = wn.synsets("location")[0]
    org_ss = wn.synsets("organization")[0]
    counter = 0
    synt_category = head
    try:
        while synt_category not in [None,'PERSON','LOC','ORG']:
            # words without meaning return empty lists and cause infinite loop, we need to raise error
            assert len(wn.synsets(synt_category))>0, f"{synt_category} has no synonyms"
            
            for ss in wn.synsets(synt_category):
                counter += 1                
                #print(ss.lemmas())
                #for hyper in ss.hypernyms():
                assert len(ss.hypernyms())>0, f"{ss} has no hypernyms"
                hyper = ss.hypernyms()[0]
                
                #print(f'for {synt_category} synonyms are: {ss}, hypernyms are: {hyper}')
                #print(f'synonym with person: {ss.wup_similarity(person_ss)}')
                #print(f'hypernym with person: {hyper.wup_similarity(person_ss)}')
                #print(f'with group: {ss.wup_similarity(group_ss)}')
                #print(f'synonym with place: {ss.wup_similarity(place_ss)}')
                #print(f'hypernym with place: {hyper.wup_similarity(place_ss)}')

                #if the syntactic similarity to one of the categories is more than 0.7, select the category

                sim_dict = {'PERSON' : ss.wup_similarity(person_ss), 'LOC': ss.wup_similarity(place_ss), 'ORG': ss.wup_similarity(org_ss)   }

                for key,value in sim_dict.items():
                    if value == max(sim_dict.values()) and value >= 0.7:
                        synt_category = key
                    else:
                        # if the synset is not similar assign the hypernym synset
                        synt_category = hyper.lemma_names()[0]
                #
                #
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAKE MAX FUNCTION FOR FOLLOWING SECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #
                """                if ss.wup_similarity(person_ss) >= 0.7:
                    synt_category = 'PERSON'
                    break
                #elif ss.wup_similarity(group_ss) >= 0.7:
                    #synt_category = 'facility'
                    #break
                elif ss.wup_similarity(place_ss) >= 0.7:
                    synt_category = 'LOC'
                    break
                elif ss.wup_similarity(org_ss) >= 0.7:
                    synt_category = 'ORG'
                    break
                else:
                    # if the synset is not similar assign the hypernym synset
                    synt_category = hyper.lemma_names()[0]"""

                #force stop at level 5 of hypernym search
                if counter == 5:
                    synt_category = None
                    break
            
    except AssertionError:
        synt_category = None
        return synt_category

    #print(f'{head} turned into a candidate {synt_category}')  
    
    return synt_category

#dictionary to assign candidate types based on named entities and part of speech tags
#the key tuple consists of (isNE, lexicographer type, plural)
cand_types_dict = {(True,'PERSON',None):'person-ne',
              (True,'NORP',None):'person-ne',
              (True,'PERSON','plural'):'person-nes',
              (True,'NORP','plural'):'person-nes',
              (False,'PERSON',None):'person-nn',
              (False,'PERSON','plural'):'person-nns',
              (True,'ORG',None):'group-ne',
              (True,'FAC',None):'group-ne',
              (False,'ORG',None):'group',
              (True,'LOC',None):'loc-ne',
              (True,'GPE',None):'loc-ne',
              (False, 'LOC',None):'loc'
            }

def get_cand_type(cand_list, cand_heads, tweet_tags):
    """
    Input: list of all noun phrases occurring in one tweet
    Output: list of pairs of np (string) and its candidate type (string) in a tuple for each np of the tweet
    """
    cand_and_type_list = []
    
    for tweet_index in tqdm(range(len(cand_list))):
        cand_types = [] 
        #tweet_candidates = [cand_list[tweet_index].split('\n\n')] if corefs else cand_list[tweet_index].split('\n\n')
        tweet_candidates = cand_list[tweet_index]#.split('\n\n')
        cand_head_type = tuple()
        for np in range(len(tweet_candidates)):
            cand_head_type = tuple() 
            rep_head = cand_heads[tweet_index][np][1][0]
            phrase_heads = cand_heads[tweet_index][np][0]

            #check if the noun phrase contains an NE tag
            isNE = False
            #print(np_pos_tags[tweet_index])
            for key in tweet_tags[tweet_index][1].keys():
                # we exclude numbered entities so "three children" are not considered named entity
                if key in tweet_candidates[np] and key not in  ['CARDINAL', 'DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY', 'ORDINAL']:
                    isNE = True                
            
            # identified entity will be none if the head is not a named entity, if it is, the NER tag will be assigned
            ner_tag = None
            
            #print(f'the head of {tweet_candidates[np]} is {rep_head}')
            
            for key in tweet_tags[tweet_index][1].keys():
                
                if rep_head in key and tweet_tags[tweet_index][1][key] not in ['CARDINAL', 'DATE', 'QUANTITY', 'TIME', 'PERCENT', 'MONEY', 'ORDINAL']:
                    ner_tag = tweet_tags[tweet_index][1][key]
                    #print(ner_tag)
    

            identified_ner = ner_tag if ner_tag != None else get_synt_category(rep_head)
                
            #print(np_pos_tags[tweet_index])
            pos_number = None
            if cand_heads[tweet_index][np][1][0] in tweet_tags[tweet_index][0].keys():
                #print(cand_heads[tweet_index][np][1][0])
                pos_tag = tweet_tags[tweet_index][0][rep_head]
                pos_number = 'plural' if pos_tag in ['NNS','NNPS'] and identified_ner in ['person','PERSON'] else None
            
            #we want to create a tuple of (is_named_entity, NE_tag/synt_category, POS-tag)
            pre_cand_type = (isNE, identified_ner, pos_number)        
            
            #print(f'{tweet_candidates[np]}\n isNE: {isNE}, ner: {identified_ner}, pos: {pos_number}')
            cand_type = cand_types_dict[pre_cand_type] if pre_cand_type in cand_types_dict.keys() else 'misc'
            
            
            cand_head_type = (tweet_candidates[np],rep_head,phrase_heads,cand_type)

            cand_types.append(cand_head_type)
        cand_and_type_list.append(cand_types)
    return cand_and_type_list





def get_cand_len(cand_list):
    # calculates number of candidates in the corpus
    sum_len = 0
    for tweet_cands in cand_list:
        sum_len += len(tweet_cands)
    return sum_len


def remove_long_nps(noun_phrase_list):
    print(f'removing long candidates...')
    initial_len = get_cand_len(noun_phrase_list)
    for tweet_id in range(len(noun_phrase_list)):
        #print(noun_phrase_list[tweet_id])
        #reverse the list of tweets nps so we avoid moving indexes and leaving out some phrases 
        for noun_p in reversed(noun_phrase_list[tweet_id]):
            i = noun_phrase_list[tweet_id].index(noun_p)
            np_split = noun_p.split()
            # Hamborg removes nps longer than 19 words, identified candidates do not seem to fulfill NP role, that's why we opt for lower threshold
            threshold = 9
            if len(np_split) > threshold:
                noun_phrase_list[tweet_id].remove(noun_phrase_list[tweet_id][i])
    len_after_removal = get_cand_len(noun_phrase_list)
    print(f'Removed {initial_len-len_after_removal} candidates longer than {threshold} words!')

    return noun_phrase_list

def remove_child_nps(noun_phrase_list):
    print(f'removing child NP candidates...')
    initial_len = get_cand_len(noun_phrase_list)
    # remove the child NPs and keep only parents, run until the sum_len stops decreasing
    after_removal_len = 0
    while after_removal_len != get_cand_len(noun_phrase_list):
        after_removal_len = get_cand_len(noun_phrase_list)
        for tweet_nps in noun_phrase_list:
            for noun_p in range(len(tweet_nps)):
                try:
                    #if the subsequent noun_p (child np) is contained in the current one, remove the child np
                    if tweet_nps[noun_p].find(tweet_nps[noun_p+1]) != -1:
                        tweet_nps.remove(tweet_nps[noun_p+1])
                        
                #ignore the error caused with end of the list
                except IndexError:
                    pass

    len_after_removal = get_cand_len(noun_phrase_list)
    print(f'Removed {initial_len-len_after_removal} child NP candidates!')
    return noun_phrase_list

def remove_char(noun_phrase_list,char):
    #removes the @ sign at the beginning of the mention
    return [[np.replace(char,'') for np in nps ] for nps in noun_phrase_list ]   