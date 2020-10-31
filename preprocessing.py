import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
import os 


def load_data(folder):
    #get current directory and input the folder to analyze
    directory_path = os.getcwd() + '/' + folder + '/'
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
    return event_df


def lemma(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

def remove_stopwords(text, stop_words):
    stopped = [word for word in text if (word not in stop_words)]
    return [word for word in stopped if (len(word) > 3)]

def preprocess_text(text):
    #remove the tagged account names
    text = text.apply(lambda x: re.sub('\@(.*?:)',' ', str(x)))
    #remove everything that is not a letter
    text = text.apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)).lower())
    text = text.apply(lambda x: word_tokenize(str(x)))
    text = text.apply(lemma)
    return text


def get_processed_data(folder):
    print('loading files...')
    event_df = load_data(folder)
    print('loaded ', event_df.shape[0], ' tweets.')
    print('preprocess tweets...')
    #select only the ones that have reached audience higher than x 
    event_df = event_df[event_df['Reach'] > 0]
    event_df["Hit_Sentence_Clean"] = preprocess_text(event_df["Hit Sentence"])
    print('removing unwanted words...')
    stop_words = stopwords.words('english')
    extra_stopwords = ['twitter','http'] 
    stop_words = list(stop_words) + extra_stopwords
    event_df["Hit_Sentence_Clean"] = event_df["Hit_Sentence_Clean"].apply(lambda x: remove_stopwords(x, stop_words))
    print('DONE! Processed ',event_df.shape[0],' tweets ready for analysis!' )
    return event_df
