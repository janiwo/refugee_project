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


def load_data(folder):
	print('loading files...')
	#get current directory and input the folder to analyze
	data_url = f"CBS - Copenhagen Business School/Kick-Ass Master Thesis - General/Data/{folder}/Raw Data/"
	directory_path = os.getcwd() + '/../../../' + data_url 
	#print(directory_path)
	event_df = pd.DataFrame()
	#append all csv files in the folder to each other
	for file_name in tqdm(glob.glob(directory_path + '*.csv')):
	    #print('Reading file: ' + file_name)
	    sheet = pd.read_csv(file_name, sep='\t', encoding = 'utf_16')
	    event_df = pd.concat([event_df,sheet],axis=0)
	#choose only relevant columns to analyze
	relevant_cols = ["Date","URL","Hit Sentence","Influencer","Country","Language","Reach",
	                 "Engagement","AVE","Sentiment","Key Phrases","Keywords","Twitter Authority", 
	                 "Tweet Id","Twitter Id","Twitter Client","Twitter Screen Name","Twitter User Profile Url",
	                 "Twitter Bio","Twitter Followers","Twitter Following","Alternate Date Format","Time","State","City"]
	event_df = event_df[relevant_cols]
	#rename columns
	event_df.rename(columns = {'Hit Sentence': 'Tweet Raw',
	    "Alternate Date Format":"Date Short",
	    "Twitter Client":"Client",
	    "Twitter Screen Name":"Screen Name",
	    "Twitter User Profile Url":"User Profile Url",
	    "Twitter Bio":"Bio",
	    "Twitter Followers":"Followers",
	    "Twitter Following":"Following"},inplace = True)

	print('loaded ', event_df.shape[0], ' tweets.')

	return event_df#.iloc[0:10000,:]


def lemma(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

def tokenize(text):
    return text.apply(lambda x: word_tokenize(str(x)))  

def remove_stopwords(text, stop_words):
    stopped = [word for word in text if (word not in stop_words)]
    return [word for word in stopped if (len(word) > 3)]

def clean_text(text):
    #remove links
    text = text.apply(lambda x: re.sub('[http|https]+://[\w\S(\.|:|/)]+',' ', str(x))) # adapted from https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
    #remove the tagged account names
    text = text.apply(lambda x: re.sub('\@(.*?:)',' ', str(x)))

    #remove everything that is not a letter and set to lowercase
    text = text.apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)).lower())

    #clean \n

    """
    TO DO:
    Clean links as well (https shows in the wordcloud)
    Progress bar (each 10% or something)
    - bigrams/trigrams to see words in connection with stopwords like NOT etc. and word that belong together (get support from theory)
    - implement and compare with the "preprocess" package
    - feature of hashtags/mentions (number of hashtags/mentions)
    - feature length of tweets (in words and characters)
    - how many overlapping words in train and test set
    - remove everything that is copied in all QTs
    
    """
    return text

def create_feature_columns(event_df):

    event_df["Hashtags"] = event_df["Tweet Raw"].apply(lambda x: re.findall('#(\w+)',str(x)))
    event_df["Mentions"] = event_df["Tweet Raw"].apply(lambda x: re.findall('@(\w+)',str(x)))
    event_df["Linked Content"] = event_df["Tweet Raw"].apply(lambda x: re.findall('[http|https]+://[\w\S(\.|:|/)]+',str(x)))
    #return True if the Tweet Raw text contains RT and QT
    event_df["Retweet"] = event_df["Tweet Raw"].astype(str).str.startswith("RT") #maybe need to improve so it looks only at the start of the string
    event_df["Quote Tweet"] = event_df["Tweet Raw"].astype(str).str.startswith("QT") #maybe need to improve so it looks only at the start of the string
    return event_df
    """
    Include RT column (boolean)
    include linked content columns
    """


def get_processed_data(event_df):

    print('preprocess tweets...')
    #select only the ones that have reached audience higher than x 
    #event_df = event_df[event_df['Reach'] > 0]

    event_df = create_feature_columns(event_df)
    preprocessor.set_options()
    event_df['Tweet Clean'] = event_df["Tweet Raw"].apply(lambda x: preprocessor.clean(x).lower())  

    event_df["Tweet Clean Tokens"] = tokenize(event_df["Tweet Clean"])
    event_df["Tweet Clean Tokens"] = event_df["Tweet Clean Tokens"].apply(lemma)

    print('removing unwanted words...')
    stop_words = stopwords.words('english')
    extra_stopwords = ['twitter','http','https','daniel','trilling','guardian','bbcnews'] 
    stop_words = list(stop_words) + extra_stopwords
    event_df["Tweet Clean Tokens"] = event_df["Tweet Clean Tokens"].apply(lambda x: remove_stopwords(x, stop_words))

    print('DONE! Processed ',event_df.shape[0],' tweets ready for analysis!' )
    return event_df


def remove_duplicate_tweets(event_df):
	event_df.reset_index(drop=True, inplace=True)

	print('removing retweets and quote tweets...')
	event_df = event_df[event_df['Quote Tweet'] != True]
	event_df = event_df[event_df['Retweet'] != True]


	print('removing duplicates...')
	event_df['tokens_to_str'] = event_df["Tweet Clean Tokens"].apply(lambda x: ' '.join(x))
	event_df.drop_duplicates(subset="tokens_to_str", inplace=True)  
	event_df.drop('tokens_to_str',axis=1,inplace=True)  
	return event_df

def fuzzy_duplicate_removal(event_df, similarity = 0.9):
	def find_matches(tweet_to_match):
	    rows = event_df['Tweet Clean'].values.tolist() #convert the df column to list

	    if (rows.index(tweet_to_match)+1) % 100 ==0:
	    	print(f'Currently on {rows.index(tweet_to_match)+1}th tweet...')

	    del rows[:rows.index(tweet_to_match)] #remove the previous rows as they have already been matched 

	    #perform similarity calculation using difflib and mark the similarities higher than threshold as duplicates
	    matches = [sm(None, tweet_to_match, row).ratio() for row in rows]
	    are_duplicates = [match>similarity for match in matches]
	    
	    return are_duplicates[1:] # return list of booleans excluding itself (index 0)

	clean_tweets_df = event_df['Tweet Clean'].to_frame() # so we can use .applymap method we only put the tweet text column (cleaned) into dataframe 
	

	fn_find_matches = lambda x: find_matches(x)
	clean_tweets_df['Duplicate']=clean_tweets_df.applymap(fn_find_matches)

	#instantiate the list where we append the locations of duplicate rows
	dup_locs = {}

	for index, row in clean_tweets_df.iterrows():
	    # the index of duplicate in the df is 1 + index + position in list of duplicates:
	    # + index since we start counting in the row in which we are checking
	    # + 1 since we exclude the similarity of the tweets with itself
	    # + np.where(clean_tweets_df['Duplicate'].iloc[index])[0] using numpy to get locations of True labels
	    dup_locs[index] = []
	    dup_locs[index].extend(index + 1 + np.where(clean_tweets_df['Duplicate'].iloc[index])[0])
	    
	    if len(dup_locs[index]) == 0:
	        del dup_locs[index]
	    #if True in clean_tweets_df['Duplicate'].iloc[index]:
	        #duplicated_rows.append(index)

	#print indexes of the fuzzily duplicate rows 
	print(f'The rows that are more than {similarity*100}% similar are: {dup_locs}')

	# first list of lists of dup_locs dictionary values is converted into a list, and then into a descending sorted set to preserve only unique values
	indexes_to_remove = sorted(set(list(pd.core.common.flatten(list(dup_locs.values())))),reverse=True)
	print(f'Removing {len(indexes_to_remove)} fuzzy duplicates')

	event_df.drop(indexes_to_remove,inplace=True)

	return event_df



from ekphrasis.classes.tokenizer import SocialTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer 
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector

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
                       "Read more &gt;&gt;&gt;"
                  ]
    for text in texts_to_remove:
        tweet = tweet.replace(text,"")
    return tweet



puncttok = nltk.WordPunctTokenizer().tokenize
social_tokenizer = SocialTokenizer(lowercase=False).tokenize
detokenizer = TreebankWordDetokenizer()

sp = SpellCorrector(corpus="english") 
seg_eng = Segmenter(corpus="english") 

# preprocessor should remove emojis and urls in the tweets
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI)

def preprocess_tweets(tweets_df):

    for twt in tqdm(range(len(tweets_df))):
        
        #clean emojis, links and remove common signatures occuring in tweets (as observed from longest candidates later on)
        tweets_df.iloc[twt] = preprocessor.clean(tweets_df.iloc[twt])
        tweets_df.iloc[twt] = remove_tweet_signatures(tweets_df.iloc[twt])
        
        
        # we are using social tokenizer due to potentially improper text structure
        #tweet = sample_df[twt].split()
        tweet = social_tokenizer(tweets_df.iloc[twt])


        #removing the irrelevant hashtags and mention using the heuristic that mentions in the beginning of the tweet 
        # and at least 2 consecutive hashtags at the end of the tweet carry no valuable information
        try:
            while tweet[0].startswith('@'):
                tweet.remove(tweet[0])
            while tweet[-1].startswith('#'):
                hashtag_end = False
                if tweet[-1].startswith('#') and tweet[-2].startswith('#'):
                    hashtag_end = True
                    tweet.remove(tweet[-1])
                elif hashtag_end:
                    tweet.remove(tweet[-1])
                else:
                    break
                    
        except IndexError:
            pass
            #sample_df.iloc[twt] = tweet


        #for hashtags that may carry information, we remove the # and split the word into more if applicable
        for word in range(len(tweet)):
            if tweet[word].startswith('#'):
                tweet[word] = tweet[word].replace('#','')
                tweet[word] = seg_eng.segment(tweet[word])

            # potentially correct spelling - but it is not working very well - corrects numbers to weird words
            #tweet[word] = sp.correct(tweet[word])

        # instead of .join we use detokenizer in order to reconstruct the cleaned sentence in a better way
        #sample_df[twt] =  " ".join(tweet) 
        tweets_df.iloc[twt] = detokenizer.detokenize(tweet)

        #  tweets that end up being empty after preprocessing will cause problems when batching, replace empty tweet with 'no_tweet_text' which we can ignore later
        tweets_df.iloc[twt] = 'no_tweet_text' if len(tweets_df.iloc[twt])==0 else tweets_df.iloc[twt]
        
    return tweets_df