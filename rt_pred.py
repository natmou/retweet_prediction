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


def non_user_agnostic_pred():

    data = pd.read_csv('tweets.csv')

    usable_cols = [u'TweetPostedTime', u'TweetID', u'TweetBody', u'TweetRetweetFlag',
                   u'TweetSource', u'TweetRetweetCount',
                   u'TweetFavoritesCount', u'TweetHashtags', u'UserID', u'UserName', u'UserScreenName',
                   u'UserLocation', u'UserDescription',
                   u'UserFollowersCount', u'UserFriendsCount', u'UserListedCount',
                   u'UserSignupDate', u'UserTweetCount', u'MacroIterationNumber']

    data = data[usable_cols].fillna('')

    encode_y = ['TweetPostedTime', 'UserLocation', 'UserSignupDate']
    encode_n = [u'UserID', u'UserFollowersCount', u'UserFriendsCount',
                u'UserListedCount', u'UserTweetCount', 'TweetRetweetCount']

    list_mapper = [(c, LabelEncoder()) for c in encode_y]
    mapper = DataFrameMapper(list_mapper)
    edf = mapper.fit_transform(data)
    edf = pd.DataFrame(edf, index = None, columns = encode_y)
    data_le = pd.concat([data[encode_n].reset_index(drop=True), edf], axis=1)
    data_le = data_le.drop_duplicates()

    X_cols = [u'UserID', 
                u'UserLocation', 
                u'UserFollowersCount', u'UserFriendsCount', u'UserListedCount',
                u'UserSignupDate', u'UserTweetCount']

    #X_cols = ['UserLocation', 'UserSignupDate']

    Y_cols = [u'TweetRetweetCount']

    X = data_le[X_cols]

    y = data_le[Y_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

    xgbmod = xgb.XGBRegressor()

    xgbmod.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)],verbose=True)

    res = xgbmod.predict(X_test)

def non_user_agnostic_pred_hashtag():

    def create_vector(row):
        sentence = row['TweetHashtags']
        vector = np.zeros(50)
        counter = 0
        for word in sentence:
            if word in model:
                vector = vector + model[word]
                counter += 1
        vector = vector/counter
        return list(vector)

    data = pd.read_csv('tweets.csv')

    usable_cols = [u'TweetPostedTime', u'TweetID', u'TweetBody', u'TweetRetweetFlag',
                   u'TweetSource', u'TweetRetweetCount',
                   u'TweetFavoritesCount', u'TweetHashtags', u'UserID', u'UserName', u'UserScreenName',
                   u'UserLocation', u'UserDescription',
                   u'UserFollowersCount', u'UserFriendsCount', u'UserListedCount',
                   u'UserSignupDate', u'UserTweetCount', u'MacroIterationNumber']

    data = data[usable_cols].fillna('')

    data['TweetHashtags'] = data.TweetHashtags.str.lower().str.split(', ')

    sentences = data.TweetHashtags.values

    model = gensim.models.Word2Vec(sentences, size=50, window=16, min_count=2, workers=5, sg=1,iter=100)

    data['vec'] = data.apply(lambda x: create_vector(x), axis=1)

     
    encode_y = ['TweetPostedTime', 'UserLocation', 'UserSignupDate']
    encode_n = [u'TweetID', u'UserID', u'UserFollowersCount', u'UserFriendsCount',
                u'UserListedCount', u'UserTweetCount', 'TweetRetweetCount', 'vec']

    list_mapper = [(c, LabelEncoder()) for c in encode_y]
    mapper = DataFrameMapper(list_mapper)
    edf = mapper.fit_transform(data)
    edf = pd.DataFrame(edf, index = None, columns = encode_y)
    data_le = pd.concat([data[encode_n].reset_index(drop=True), edf], axis=1)

    explode = pd.DataFrame(data_le.vec.values.tolist())
    explode['TweetID'] = data_le.TweetID.values

    data_le = pd.merge(data_le, explode, on='TweetID')
    del data_le['vec']

    Y_cols = [u'TweetRetweetCount']
    X_cols = list(set(data_le.columns) - set(Y_cols))
    #X_cols = ['UserLocation', 'UserSignupDate']

    X = data_le[X_cols]

    y = data_le[Y_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

    xgbmod = xgb.XGBRegressor()

    xgbmod.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)],verbose=True)

    res = xgbmod.predict(X_test)

def word2vec_test():
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

    stoplist = stopwords.words('english')

    data = pd.read_csv('tweets.csv')
    data = data[['TweetHashtags', 'TweetBody']].fillna('')
    data = data.drop_duplicates()

    data['TweetBody'] = data.TweetBody.str.lower().str.replace(r'[^A-Za-z0-9-" "]+', '') 
    data['TweetBody'] = data.apply(lambda x: nltk.word_tokenize(x['TweetBody']), axis=1)
    data['TweetBody'] = data.apply(lambda x: [word for word in x['TweetBody'] 
                                                    if word not in stoplist 
                                                    and len(word)>1], axis=1) 

    sentences_body = data.TweetBody.values
    model_body = gensim.models.Word2Vec(sentences_body, size=50, window=4, min_count=2, workers=5, sg=1,iter=50)
    data['vec_body'] = data.apply(lambda x: create_vector(x['TweetBody'], model_body), axis=1)

    data['TweetHashtags'] = data.TweetHashtags.str.lower().str.split(', ')

    sentences_hashtag = data.TweetHashtags.values
    model_hashtag = gensim.models.Word2Vec(sentences_hashtag, size=50, window=16, min_count=2, workers=5, sg=1,iter=50)
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

    res = xgbmod.predict(X_test)



'''TweetPostedTime              42368
TweetID                      42368
TweetBody                    42368
TweetRetweetFlag             42368
TweetSource                  42368
TweetInReplyToStatusID         101
TweetInReplyToUserID           189
TweetInReplyToScreenName       189
TweetRetweetCount            42368
TweetFavoritesCount          42368
TweetHashtags                42268
TweetPlaceID                  1000
TweetPlaceName                1000
TweetPlaceFullName            1000
TweetCountry                   999
TweetPlaceBoundingBox         1000
TweetPlaceAttributes             0
TweetPlaceContainedWithin        0
UserID                       42368
UserName                     42368
UserScreenName               42368
UserLocation                 26342
UserDescription              38004
UserLink                     16599
UserExpandedLink             16562
UserFollowersCount           42368
UserFriendsCount             42368
UserListedCount              42368
UserSignupDate               42368
UserTweetCount               42368
MacroIterationNumber         42368
tweet.place                   1000'''