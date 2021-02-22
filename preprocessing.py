import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
import os 
import preprocessor as p


def load_data(folder):
	print('loading files...')
	#get current directory and input the folder to analyze
	data_url = f"CBS - Copenhagen Business School/Kick-Ass Master Thesis - General/Data/{folder}/Raw Data/"
	directory_path = os.getcwd() + '/../../../' + data_url 
	print(directory_path)
	event_df = pd.DataFrame()
	#append all csv files in the folder to each other
	for file_name in glob.glob(directory_path + '*.csv'):
	    print('Reading file: ' + file_name)
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

	return event_df#.iloc[0:1000,:]


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
    event_df["Retweet"] = event_df["Tweet Raw"].apply(lambda x: re.search('RT',str(x))!= None) #maybe need to improve so it looks only at the start of the string
    event_df["Quote Tweet"] = event_df["Tweet Raw"].apply(lambda x: re.search('QT',str(x)) != None) #maybe need to improve so it looks only at the start of the string
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

    event_df['Tweet Clean'] = event_df["Tweet Raw"].apply(lambda x: p.clean(x).lower())  

    event_df["Tweet Clean Tokens"] = tokenize(event_df["Tweet Clean"])
    event_df["Tweet Clean Tokens"] = event_df["Tweet Clean Tokens"].apply(lemma)

    print('removing unwanted words...')
    stop_words = stopwords.words('english')
    extra_stopwords = ['twitter','http','https','daniel','trilling'] 
    stop_words = list(stop_words) + extra_stopwords
    event_df["Tweet Clean Tokens"] = event_df["Tweet Clean Tokens"].apply(lambda x: remove_stopwords(x, stop_words))

    print('DONE! Processed ',event_df.shape[0],' tweets ready for analysis!' )
    return event_df
