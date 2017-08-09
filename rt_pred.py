"""
Intro
"""

from __future__ import division
import pandas as pd
import numpy as np
import gensim
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper

def body_model(data):
    stoplist = stopwords.words('english')
    body = (data.TweetBody.fillna('')
                             .drop_duplicates()
                             .str.lower()
                             .str.replace(r'[^A-Za-z0-9-" "]+', ''))

    body = body.apply(lambda x: nltk.word_tokenize(x))
    body = body.apply(lambda x: [word for word in x
                                        if word not in stoplist 
                                        and len(word)>1]) 
    sentences_body = body.values
    model = gensim.models.Word2Vec(sentences_body, size=50, window=4, min_count=2, workers=5, sg=1,iter=50)
    return model

def hashtag_model(data):
    hashtag = (data['TweetHashtags'].drop_duplicates()
                                    .fillna('')
                                    .str.lower().str.split(', '))
    sentences_hashtag = hashtag.values
    model = gensim.models.Word2Vec(sentences_hashtag, size=50, window=16, min_count=2, workers=5, sg=1,iter=50)
    return model

def create_vector(col, model):
    sentence = col
    vector = np.zeros(50)
    counter = 0
    for word in sentence:
        if word in model:
            vector = vector + model[word]
            counter += 1
    vector = vector/counter
    return list(vector)

def non_user_agnostic_pred(model_hashtag, model_body, data):

    stoplist = stopwords.words('english')

    usable_cols = [u'TweetPostedTime', u'TweetID', u'TweetBody', u'TweetRetweetFlag',
                   u'TweetSource', u'TweetRetweetCount',
                   u'TweetFavoritesCount', u'TweetHashtags', u'UserID', u'UserName', u'UserScreenName',
                   u'UserLocation', u'UserDescription',
                   u'UserFollowersCount', u'UserFriendsCount', u'UserListedCount',
                   u'UserSignupDate', u'UserTweetCount', u'MacroIterationNumber']

    data = data[usable_cols].fillna('')

    data['TweetHashtags'] = data.TweetHashtags.str.lower().str.split(', ')
    data['hashtag_vec'] = data.apply(lambda x: create_vector(x['TweetHashtags'], model_hashtag), axis=1)

    data['TweetBody'] = data.TweetBody.str.lower().str.replace(r'[^A-Za-z0-9-" "]+', '') 
    data['TweetBody'] = data.apply(lambda x: nltk.word_tokenize(x['TweetBody']), axis=1)
    data['TweetBody'] = data.apply(lambda x: [word for word in x['TweetBody'] 
                                                    if word not in stoplist 
                                                    and len(word)>1], axis=1) 

    data['body_vec'] = data.apply(lambda x: create_vector(x['TweetBody'], model_body), axis=1)    

    encode_y = ['TweetPostedTime', 'UserLocation', 'UserSignupDate']
    encode_n = [u'TweetID', u'UserID', u'UserFollowersCount', u'UserFriendsCount',
                u'UserListedCount', u'UserTweetCount', 'TweetRetweetCount', 'hashtag_vec', 'body_vec']

    list_mapper = [(c, LabelEncoder()) for c in encode_y]
    mapper = DataFrameMapper(list_mapper)
    edf = mapper.fit_transform(data)
    edf = pd.DataFrame(edf, index = None, columns = encode_y)
    data_le = pd.concat([data[encode_n].reset_index(drop=True), edf], axis=1)

    explode_body = pd.DataFrame(data_le.body_vec.values.tolist())
    explode_body.columns = np.arange(0,50).astype('str')
    explode_body['TweetID'] = data_le.TweetID.values

    explode_hashtag = pd.DataFrame(data_le.hashtag_vec.values.tolist())
    explode_hashtag.columns = np.arange(0,50).astype('str')
    explode_hashtag['TweetID'] = data_le.TweetID.values

    data_le = pd.merge(data_le, explode_body, on='TweetID', how = 'inner')
    data_le = pd.merge(data_le, explode_hashtag, on='TweetID', how = 'inner')
    del data_le['body_vec']
    del data_le['hashtag_vec']

    Y_cols = [u'TweetRetweetCount']
    X_cols = list(set(data_le.columns) - set(Y_cols))
    #X_cols = ['UserLocation', 'UserSignupDate']

    X = data_le[X_cols]

    y = data_le[Y_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

    xgbmod = xgb.XGBRegressor()

    xgbmod.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)],verbose=True)

    return {'mean' : np.mean(y_train.values),
            'std' : np.std(y_train.values),
            'median' : np.median(y_train.values),
            'max' : np.max(y_train.values),
            'min' : np.min(y_train.values),
            'rmse' : xgbmod.best_score,
            'size' : len(y_test.values)}

def text_based_prediction(model_hashtag, model_body, data):

    stoplist = stopwords.words('english')
    data = data[['TweetHashtags', 'TweetBody', 'TweetRetweetCount']].fillna('')
    data = data.drop_duplicates()

    data['TweetBody'] = data.TweetBody.str.lower().str.replace(r'[^A-Za-z0-9-" "]+', '') 
    data['TweetBody'] = data.apply(lambda x: nltk.word_tokenize(x['TweetBody']), axis=1)
    data['TweetBody'] = data.apply(lambda x: [word for word in x['TweetBody'] 
                                                    if word not in stoplist 
                                                    and len(word)>1], axis=1) 

    data['vec_body'] = data.apply(lambda x: create_vector(x['TweetBody'], model_body), axis=1)

    data['TweetHashtags'] = data.TweetHashtags.str.lower().str.split(', ')

    data['vec_hashtag'] = data.apply(lambda x: create_vector(x['TweetHashtags'], model_hashtag), axis=1)

    explode_hashtag = pd.DataFrame(data.vec_hashtag.values.tolist())
    explode_hashtag.columns = np.arange(0,50)

    explode_body = pd.DataFrame(data.vec_body.values.tolist())
    explode_body.columns = np.arange(50,100)

    X = pd.concat([explode_body, explode_hashtag], axis=1, join_axes=[explode_body.index])
    y = data['TweetRetweetCount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

    xgbmod = xgb.XGBRegressor()
    xgbmod.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)],verbose=True)

    return {'mean' : np.mean(y_train.values),
            'std' : np.std(y_train.values),
            'median' : np.median(y_train.values),
            'max' : np.max(y_train.values),
            'min' : np.min(y_train.values),
            'rmse' : xgbmod.best_score,
            'size' : len(y_test.values)}

if __name__ == '__main__':
    data = pd.read_csv('tweets.csv')
    body_mod = body_model(data)
    hasht_mod = hashtag_model(data)
    result_text = text_based_prediction(hasht_mod, body_mod, data)
    result_user = non_user_agnostic_pred(hasht_mod, body_mod, data) 
    print 'results for prediction based on text and hashtags only :'
    print result_text
    print 'result for preduction based on text and users features (non user agnostic):'
    print result_user