
from stanza.server import CoreNLPClient
from tqdm import tqdm
#import candidate_preprocessing as cand_prep


# get noun phrases with tregex
def get_noun_phrases(tweet, client, annotators=None):
    """
    Input: client = CoreNLPClient instance
           tweet = tweet text
           annotators = allowed CoreNLP operations
    Output: list of all noun phrases in the tweet
    """
    pattern = 'NP'
    matches = client.tregex(tweet,pattern,annotators=annotators)

    return [sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]


def get_coref_chains(tweet,client,annotators=None):

    ann = client.annotate(tweet)        
    tweet_chains = ann.corefChain
    all_chains = list()
    
    
    for chain in tweet_chains:
        chain_words = list()
        # Loop through every mention of this chain
        for mention in chain.mention:
            # Get the sentence in which this mention is located, and get the words which are part of this mention
            words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
            #build a string out of the words of this mention
            ment_word = ' '.join([word.word for word in words_list])
            
            chain_words.append(ment_word)
            
        #the corefering words will be stored alongside the index of their representative in a tuple
        coref_group = (chain_words,chain.representative)
        #coref_cand = coref_group[0][coref_group[1]]
        all_chains.append(coref_group)


    return all_chains




def extract_candidates(event_df, all=True):

    corefs_list = list()
    tweets_list = list(event_df)

    #so we have control over whether we extract only np or coref candidates
    #nps = True if all == True or all == 'nps' else False
    #corefs = True if all == True or all == 'corefs' else False

    with CoreNLPClient(annotators=["tokenize,ssplit,pos,lemma,parse,coref,ner,depparse"], properties ={'coref.algorithm' : 'statistical'}, timeout=600000, memory='1G') as client:

        # get noun phrases with tregex using get_noun_phrases function
        print('extracting noun phrases...')
        tqdm.pandas()
        noun_phrase_list = list(event_df.progress_apply(get_noun_phrases,args=(client,"tokenize,ssplit,pos,lemma,parse")))
        #noun_phrase_list = [get_noun_phrases(client,tweets_list[tweet_id], annotators="tokenize,ssplit,pos,lemma,parse") for tweet_id in tqdm(range(len(tweets_list)))]


        #print('extracting coreference chains...')
        # get coreference chains using the .annotate method of client handled by get_coref_chain function  

        #corefs_list = list(event_df.progress_apply(get_coref_chains,args=(client,)))
        #for tweet_id in tqdm(range(len(tweets_list))):
            #coref_chains = [chain for chain in get_coref_chain(event_df[tweet_id],client)] 

        #corefs_list.append(['no_candidate']) if len(corefs_list) == 0 else corefs_list.append(coref_chains)
                
             

    return noun_phrase_list#, corefs_list

def extract_corefs(event_df, all=True):

    #corefs_list = list()
    #tweets_list = list(event_df)

    #so we have control over whether we extract only np or coref candidates
    #nps = True if all == True or all == 'nps' else False
    #corefs = True if all == True or all == 'corefs' else False

    with CoreNLPClient(annotators=["tokenize,ssplit,pos,lemma,parse,coref,ner,depparse"], properties ={'coref.algorithm' : 'neural'}, timeout=600000, memory='8G') as client:

        def get_coref_chains(tweet,client=client):

            ann = client.annotate(tweet)        
            tweet_chains = ann.corefChain
            all_chains = list()
            
            
            for chain in tweet_chains:
                chain_words = list()
                # Loop through every mention of this chain
                for mention in chain.mention:
                    # Get the sentence in which this mention is located, and get the words which are part of this mention
                    words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                    #build a string out of the words of this mention
                    ment_word = ' '.join([word.word for word in words_list])
                    
                    chain_words.append(ment_word)
                    
                #the corefering words will be stored alongside the index of their representative in a tuple
                coref_group = (chain_words,chain.representative)
                #coref_cand = coref_group[0][coref_group[1]]
                all_chains.append(coref_group)


            return all_chains
        # get noun phrases with tregex using get_noun_phrases function
        #print('extracting noun phrases...')
        tqdm.pandas()
        #noun_phrase_list = list(event_df.progress_apply(get_noun_phrases,args=(client,"tokenize,ssplit,pos,lemma,parse")))
        #noun_phrase_list = [get_noun_phrases(client,tweets_list[tweet_id], annotators="tokenize,ssplit,pos,lemma,parse") for tweet_id in tqdm(range(len(tweets_list)))]


        print('extracting coreference chains...')
        # get coreference chains using the .annotate method of client handled by get_coref_chain function  

        corefs_list = list(event_df.progress_apply(get_coref_chains))
        #for tweet_id in tqdm(range(len(tweets_list))):
            #coref_chains = [chain for chain in get_coref_chain(event_df[tweet_id],client)] 

        #corefs_list.append(['no_candidate']) if len(corefs_list) == 0 else corefs_list.append(coref_chains)
                
             

    return corefs_list

def candidate_identification(tweet_series, stanza_pipeline, batch_size):

    import stanza
    from stanza_batch import batch
    from nltk.tokenize import sent_tokenize


    all_tweets_list = list(tweet_series) 

    print('annotating the tweet corpus...')
    tagged_tweets = [tweet for tweet in tqdm(batch(all_tweets_list, stanza_pipeline, batch_size=batch_size))] 


    noun_phrase_list = extract_candidates(tweet_series) #, corefs_list


    return noun_phrase_list,   tagged_tweets # corefs_list,

    #return noun_phrase_list if nps == True
    #return coref_chains if corefs == True


"""coref format = (['word1','word2','word3'], 1)
   

#pick out only the representative mention as the candidate's rep. phrase
for tweet_id in corefs_list:
        tw_corefs = [coref[0][coref[1]] for coref in coref_chains] 
        # empty list would cause problems in the following steps, that is why we append 'no_candidate' to empty lists
        corefs_list.append(tw_corefs) if len(tw_corefs) != 0 else corefs_list.append(['no_candidate'])

corefs_list  = [[crf.replace('@','').lower() for crf in crfs ] for crfs in corefs_list ]     
corefs_list"""