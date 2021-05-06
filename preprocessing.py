import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
import os 
import numpy as np
import preprocessor
from difflib import SequenceMatcher as sm
from tqdm import tqdm
from ekphrasis.classes.tokenizer import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer 
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector



def fuzzy_duplicate_removal(event_df, similarity = 0.7):
    # https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536
    # 70,000 tweets take ~400 seconds
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from scipy.sparse import csr_matrix
    import sparse_dot_topn.sparse_dot_topn as ct
    import time

    def ngrams(string,n=3):
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def awesome_cossim_top(A, B, ntop, lower_bound=0):
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape
     
        idx_dtype = np.int32
     
        nnz_max = M*ntop
     
        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)
        ct.sparse_dot_topn(
                M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop,
                lower_bound,
                indptr, indices, data)
        return csr_matrix((data,indices,indptr),shape=(M,N))

    dupl_removed = event_df.copy()
    # first, remove tweets that are 100% similar (lowercased)
    print(f'Tweets at the start: {dupl_removed.shape[0]}')    
    dupl_removed['is_dup'] = dupl_removed['text_clean'].duplicated()
    dupl_removed = dupl_removed[dupl_removed['is_dup']==False]
    print(f'Tweets after 100% duplicates removed: {dupl_removed.shape[0]}') 

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(dupl_removed['text_clean'])

    print('calculating similarities across documents...')

    t0 = time.time()
    matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, similarity)
    t = time.time()-t0
    print(f"Similarity calculation completed in {t} seconds")


    print('removing fuzzy duplicates...')
    indices_to_remove = set()
    indices_to_protect = set()
    for row, col in tqdm(zip(*matches.nonzero())):
        val = matches[row, col]
        if row !=col:
            indices_to_protect.add(row)
            if col not in indices_to_protect:
                indices_to_remove.add(col)
    
    
    for i in indices_to_remove:
        dupl_removed['is_dup'][i] = True
    #tweet_corpus = pd.Series(tweet_corpus)
    #tweet_corpus.drop(indices_to_remove,inplace=True)            
    #tweet_corpus.reset_index(drop=True, inplace=True)
    fuzzy_removed = dupl_removed[dupl_removed['is_dup']==False]
    print(f'{fuzzy_removed.shape[0]} tweets left after {similarity*100}% similar tweets (by cosine similarity) removed')

    return fuzzy_removed


def remove_tweet_signatures(tweet):
    """
    Frequently occuring text and tweet signatures should be removed
    
    Input: full tweet text
    Output: tweet - the strings in the list
    """
    texts_to_remove = ["Greece has a deadly new migration policy and all of Europe is to blame",
                       "| The Guardian",
                       "| Photo via Evening Standard",
                       "| Greece",
                       "| DW News ",
                       "- @WashTimes",
                       "(Guardian) Story:",
                       "| Daniel Trilling",
                       " | Globaldevelopment",
                       "| Global development: sant",
                       "via @TheNationalUAE",
                       "Read more &gt;&gt;&gt;",
                       "@YahooNews | #Ethiopia |",
                       "via @yahooNewsUK"
                       "via @YahooNews",
                       "via @Yahoo",
                       "via @FoxNewsPolitics"
                       "| FoxNews"
                  ]
    for text in texts_to_remove:
        tweet = tweet.replace(text,"")
    return tweet



#puncttok = nltk.WordPunctTokenizer().tokenize


#more advanced tokenizer gives freedom to adjust the way tokens are split
social_pipeline = ["TAG", "EMAIL", "USER", "HASHTAG", "CASHTAG", "PHONE", "PERCENT", "NUMBER","WORD"]
tokenizer = Tokenizer(pipeline = social_pipeline, lowercase=False).tokenize
detokenizer = TreebankWordDetokenizer()

spell_cor = SpellCorrector(corpus="english") 
seg_eng = Segmenter(corpus="english") 

# preprocessor should remove emojis and urls in the tweets
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI)

def preprocess_tweets(tweet):


    #clean emojis, links and remove common signatures occuring in tweets (as observed from longest candidates later on)
    tweet = preprocessor.clean(tweet)
    #tweet = remove_tweet_signatures(tweet)
    
    
    # we are using social tokenizer due to potentially improper text structure
    #tweet = tweet.split()
    tweet = tokenizer(tweet)
    
    #removing the irrelevant hashtags and mention using the heuristic that mentions in the beginning of the tweet 
    # and at least 2 consecutive hashtags at the end of the tweet carry no valuable information
    try:
        while tweet[0].startswith('@'):
            tweet.remove(tweet[0])

        if tweet[-1].startswith('@') and tweet[-2].startswith('@'):
            while tweet[-1].startswith('@'):
                tweet.remove(tweet[-1])

        if tweet[-1].startswith('#') and tweet[-2].startswith('#'):
            while tweet[-1].startswith('#'):
                tweet.remove(tweet[-1])
                
    except IndexError:
        pass
        #sample_df.iloc[twt] = tweet


    #for hashtags that may carry information, we remove the # and split the word into more if applicable
    for word in range(len(tweet)):
        if tweet[word].startswith('#'):
            tweet[word] = tweet[word].replace('#','')
            tweet[word] = seg_eng.segment(tweet[word])

        # potentially correct spelling - but it is not working very well - corrects numbers to weird words
        #tweet[word] = spell_cor.correct(tweet[word])

    # instead of .join we use detokenizer in order to reconstruct the cleaned sentence in a better way
    #sample_df[twt] =  " ".join(tweet) 
    tweet = detokenizer.detokenize(tweet)
    
    
    #  tweets that end up being empty after preprocessing will cause problems when batching, replace empty tweet with 'no_tweet_text' which we can ignore later
    tweet = 'no_tweet_text' if len(tweet)==0 else tweet
    return tweet