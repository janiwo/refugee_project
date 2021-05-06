
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

    return [sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence if len(sentence[match_id]['spanString'].split())<5 ]


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


def replace_corefs(tweet_series, all=True):
    
    from stanza.server import CoreNLPClient
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer 

    detokenize = TreebankWordDetokenizer().detokenize
    
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','parse',"coref"], 
                       properties ={'coref.algorithm' : 'neural','ssplit':'eolonly'}, 
                       timeout=600000, memory='8G') as client:

        def resolve_corefs(tweet,client=client):

            ann = client.annotate(tweet)        
            tweet_chains = ann.corefChain
            all_chains = list()
            all_locs = list()
            #print(tweet)
            
            for chain in tweet_chains:
                chain_words = list()
                word_locs = list()
                # Loop through every mention of this chain
                for mention in chain.mention:
                    # Get the sentence in which this mention is located, and get the words which are part of this mention
                    words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                    #build a string out of the words of this mention
                    coref_mention = ' '.join([word.word for word in words_list])
                    identified_mention_loc = (mention.sentenceIndex,mention.beginIndex,mention.endIndex)
                    
                    chain_words.append(coref_mention)
                    word_locs.append(identified_mention_loc)
                    
                #the corefering words will be stored alongside the index of their representative in a tuple
                coref_group = (chain_words,chain.representative)
                #coref_cand = coref_group[0][coref_group[1]]
                all_chains.append(coref_group)
                all_locs.append(word_locs)
            
            #print(all_locs)
            #print(all_chains)
            tweet = sent_tokenize(tweet)
            for sent_id in range(len(tweet)):
                tweet[sent_id]=word_tokenize(tweet[sent_id])
            #print(tweet)
            for coref_words,chain_locs in zip(all_chains,all_locs):
                #print(coref,lc)
                rep_mention_id = coref_words[1]
                rep_mention = coref_words[0][rep_mention_id]
                for word,loc in zip(coref_words[0],chain_locs):
                    tweet[loc[0]][loc[1]:loc[2]] = [rep_mention]
                    #print(tweet)

            for sent_id in range(len(tweet)):
                tweet[sent_id] = detokenize(tweet[sent_id])
                #print(tweet[sent_id])
                
                
            tweet = detokenize(tweet)  
                
            #tweet = [detokenize(sent) for sents in tweet for sent in detokenize(sents)]
            #print(tweet)
            return tweet
        
        
        def tokenizer(tweet):
            tweet = word_tokenize(tweet)
            tweet = ' '.join(tweet)
            tweet = sent_tokenize(tweet)
            tweet = '\n'.join(tweet)
            return tweet
        # get noun phrases with tregex using get_noun_phrases function
        #print('extracting noun phrases...')
        tqdm.pandas()
        #noun_phrase_list = list(event_df.progress_apply(get_noun_phrases,args=(client,"tokenize,ssplit,pos,lemma,parse")))
        #noun_phrase_list = [get_noun_phrases(client,tweets_list[tweet_id], annotators="tokenize,ssplit,pos,lemma,parse") for tweet_id in tqdm(range(len(tweets_list)))]


        print('preparing coreference chains...')
        # get coreference chains using the .annotate method of client handled by get_coref_chain function  
        tweet_series = tweet_series.progress_apply(tokenizer)
        
        print('resolving coreference chains...')    
        corefs_series = tweet_series.progress_apply(resolve_corefs)            

    return corefs_series

def extract_candidates(event_df, all=True):

    corefs_list = list()
    tweets_list = list(event_df)

    #so we have control over whether we extract only np or coref candidates
    #nps = True if all == True or all == 'nps' else False
    #corefs = True if all == True or all == 'corefs' else False

    with CoreNLPClient(annotators=["tokenize,ssplit,pos,parse"], timeout=600000, memory='8G') as client:

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
                word_locs = list()
                # Loop through every mention of this chain
                for mention in chain.mention:
                    # Get the sentence in which this mention is located, and get the words which are part of this mention
                    words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
                    #build a string out of the words of this mention
                    ment_word = ' '.join([word.word for word in words_list])
                    
                    identified_words_loc = (mention.sentenceIndex,mention.beginIndex,mention.endIndex)
                    chain_words.append(ment_word)
                    word_locs.append(identified_words_loc)
                    
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