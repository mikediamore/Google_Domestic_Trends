#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:39:07 2017

@author: michael
"""

import pandas as pd
import urllib
import statsmodels.api as sm
import numpy as np
import quandl
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras.api.keras.losses as k
import pdb

quandl_tkn = "reqa36mksyfgx9BR6r88" #quandl_token
vol_scaler = 10**5
feature_scaler = 10**5


def get_google_trends(gindex, start_date=20040101):
    """
    Retrives URL data from Google_trends

    Available Indices:

       'ADVERT', 'AIRTVL', 'AUTOBY', 'AUTOFI', 'AUTO', 'BIZIND', 'BNKRPT',
       'COMLND', 'COMPUT', 'CONSTR', 'CRCARD', 'DURBLE', 'EDUCAT', 'INVEST',
       'FINPLN', 'FURNTR', 'INSUR', 'JOBS', 'LUXURY', 'MOBILE', 'MTGE',
       'RLEST', 'RENTAL', 'SHOP', 'TRAVEL', 'UNEMPL'
    """
    base = 'http://finance.google.com/finance/historical?q=GOOGLEINDEX_US:'
    full = '%s%s&output=csv&startdate=%s' % (base, gindex, start_date)
    df = pd.read_csv(urllib.request.urlopen(full),index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df['Close']
    return(df)


def get_indices_df(indices,start_date=20040101):
    my_dict = {}
    for index in indices:
        my_dict[index] = get_google_trends(index,start_date=start_date)
    my_df = pd.DataFrame.from_dict(my_dict)
    return(my_df)

def get_eod_prices(ticker,start_date,end_date):
    if start_date is None and end_date is None:
        eod_price_df = quandl.get("EOD/" + ticker,authtoken=quandl_tkn,
                                      start_date=start_date)
    elif end_date is None: #Grabs from specified beginning to unspecified end
        eod_price_df = quandl.get("EOD/" + ticker,authtoken=quandl_tkn,
                              start_date=start_date)
    else:
        eod_price_df = quandl.get("EOD/" + ticker,authtoken=quandl_tkn, 
                                  start_date=start_date,end_date=end_date)
        eod_price_df = eod_price_df
    eod_price_df = eod_price_df[['Adj_Open','Adj_High','Adj_Low','Adj_Close']]
    return(eod_price_df)

def convert_to_rets(eod_prices):
    return (eod_prices.pct_change())

def make_stationary(my_df):
    my_df = my_df.apply(np.log,axis=1)
    return(my_df)
    
def normalize_ts(series,lookback):
    rolling_mean = series.rolling(window=lookback).mean()
    rolling_std = series.rolling(window=lookback).std() + .10**-5
    normalized = (series-rolling_mean)/rolling_std
    normalized = normalized.dropna()
    return(normalized)

def volatility_estimate(Op,Hi,Lo,Cl):
    u = np.log(Hi/Op)
    d = np.log(Lo/Op)
    c = np.log(Cl/Op)
    sig = 0.511*(u-d)**2-0.019*(c*(u+d)-2*(u*d))-0.383*c**2
    return(sig)

indices = ['ADVERT', 'AIRTVL', 'AUTOBY', 'AUTOFI', 'AUTO', 'BIZIND', 'BNKRPT',
       'COMLND', 'COMPUT', 'CONSTR', 'CRCARD', 'DURBLE', 'EDUCAT', 'INVEST',
       'FINPLN', 'FURNTR', 'INSUR', 'JOBS', 'LUXURY', 'MOBILE', 'MTGE',
       'RLEST', 'RENTAL', 'SHOP', 'TRAVEL', 'UNEMPL']
#
#Preprocessor
my_df = get_indices_df(indices)
my_df = my_df.bfill()
start_date = my_df.index[0]
end_date = my_df.index[-1]
spy_prices = get_eod_prices('SPY',start_date, end_date)
spy_rets = convert_to_rets(spy_prices)
spy_vol = volatility_estimate(spy_prices['Adj_Open'],spy_prices['Adj_High'],spy_prices['Adj_Low'],spy_prices['Adj_Close'])*vol_scaler
my_df = my_df.ix[spy_vol.index,:]
                             
#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(my_df,spy_vol,test_size=0.25,shuffle=False)
X_train_normalized = X_train.apply(lambda x: normalize_ts(x,10),axis=1)*feature_scaler
X_test_normalized = X_test.apply(lambda x: normalize_ts(x,10),axis=1)*feature_scaler


def setup_tensor(size):
    W1 = tf.Variable(tf.truncated_normal(shape=[size[0],size[1]]))
    b1 = tf.Variable(tf.zeros(shape=size[1]))
    return (W1,b1)

def rnn_model(name,n_hidden,x):
    with tf.variable_scope(name):
        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden) #tweak forget gate proba
        x = tf.unstack(x,axis=1)
        outputs,state = tf.contrib.rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return (outputs,state)

def output_activations(prev_out,weights,biases):
    return(tf.matmul(prev_out,weights)+biases)

def my_GRU():
    return (tf.contrib.rnn.GRUCell(hidden_size,activation=tf.tanh))

#Taken from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#Start of tf model
hidden_size = 128 #LSTM
n_units = 1024 #Dense
batch_size = 32
timesteps = 10
num_input = X_train_normalized.shape[1]
epochs = 25
num_layers = 10

#Reformat the data to have a lookback of 10, which means creating a new dataframe of size
X_train_rebatched = series_to_supervised(X_train_normalized,n_in=timesteps-1)
y_train = y_train[X_train_rebatched.index].astype(np.float32)
#y_train = np.array(y_train)


X_test_rebatched = series_to_supervised(X_test_normalized,n_in=timesteps-1)
y_test = y_test[X_test_rebatched.index].astype(np.float32)

X_train_rebatched = np.array(X_train_rebatched).reshape(-1,timesteps,num_input).astype(np.float32)
X_test_rebatched = np.array(X_test_rebatched).reshape(-1,timesteps,num_input).astype(np.float32)
num_batches = int(np.round(len(X_train_rebatched)/batch_size))

test_batch_size = len(X_test_rebatched)
num_test_batches = int(np.round(len(X_test_rebatched)/test_batch_size))

import keras
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten

def GRU_Model(X,hidden_units,timesteps,):
    input_ = Input(shape=(timesteps,X.shape[2]),name='X_in')
    layer1 = GRU(hidden_units,return_sequences=True)(input_)
    layer2 = GRU(hidden_units,return_sequences=True)(layer1)
    layer3 = GRU(hidden_units,return_sequences=True)(layer2)
    layer4 = GRU(hidden_units,return_sequences=True)(layer3)
    layer5 = GRU(hidden_units,return_sequences=True)(layer4)
    layer6 = GRU(hidden_units,return_sequences=True)(layer5)
    layer7 = GRU(hidden_units,return_sequences=True)(layer6)
    layer8 = GRU(hidden_units,return_sequences=True)(layer7)
    layer9 = GRU(hidden_units)(layer8)
    dense = Dense(1024,activation='linear')(layer9)
    output = Dense(1,activation='linear')(dense)
    
    my_model = Model(input_, output=output)
    optimizer_adam = keras.optimizers.adam(lr=0.0002) 
    my_model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer_adam, metrics=['mse'])
    
    return (my_model)


def DenselyConnected_X(X_train,GRU_units,num_units):
        #shape=(X_train.shape[1],1)
        #batch_shape=(X_train.shape[1],1)
        X_train_input = Input(shape=(timesteps,X_train.shape[2]),name='orig')
        network = TimeDistributed(Dense(num_units,activation='relu'))(X_train_input)
        network = BatchNormalization()(network)
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = Dropout(.3)(network)
        
        
        # CONV-RELU-POOL 2
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network =TimeDistributed( Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = Dropout(.3)(network)

        
        # CONV-RELU-POOL 3
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = Dropout(.3)(network)

        
        # CONV-RELU-POOL 4
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = TimeDistributed(Dense(num_units,activation='relu'))(network)
        network = BatchNormalization()(network)
        network = Dropout(.3)(network)
        
        
#         network = GRU(units=GRU_units,return_sequences=True)(network)
#         network = BatchNormalization()(network)
#         network = Dropout(.3)(network)
        
        gru_1 = GRU(GRU_units, return_sequences=True, name='gru1')(network)
        gru_1b = GRU(GRU_units, return_sequences=True, name='gru1_b')(network)
        gru1_merged = keras.layers.merge([gru_1, gru_1b],mode='concat',concat_axis=1)
        gru_2 = GRU(GRU_units, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(GRU_units, return_sequences=True, name='gru2_b')(gru1_merged)
        network = keras.layers.merge([gru_2,gru_2b],mode='concat',concat_axis=1)
        network = Dropout(.3)(network)
        
        gru_1 = GRU(GRU_units, return_sequences=True, name='grua1')(network)
        gru_1b = GRU(GRU_units, return_sequences=True,  name='grua1_b')(network)
        gru1_merged = keras.layers.merge([gru_1, gru_1b],mode='concat',concat_axis=1)
        gru_2 = GRU(GRU_units, return_sequences=True,  name='grub2')(gru1_merged)
        gru_2b = GRU(GRU_units, return_sequences=True, name='grub2_b')(gru1_merged)
        network = keras.layers.merge([gru_2,gru_2b],mode='concat',concat_axis=1)
        network = Dropout(.3)(network)
        
        gru_1 = GRU(GRU_units, return_sequences=True,  name='gru1a')(network)
        gru_1b = GRU(GRU_units, return_sequences=True, name='gru1_ba')(network)
        gru1_merged = keras.layers.merge([gru_1, gru_1b],mode='concat',concat_axis=1)
        gru_2 = GRU(GRU_units, return_sequences=True,  name='gru2a')(gru1_merged)
        gru_2b = GRU(GRU_units, return_sequences=True,  name='gru2_ba')(gru1_merged)
        network = keras.layers.merge([gru_2,gru_2b],mode='concat',concat_axis=1)
        network = Dropout(.3)(network)
        
        gru_1 = GRU(GRU_units, return_sequences=True, name='grua1a')(network)
        gru_1b = GRU(GRU_units, return_sequences=True, name='grua1_ba')(network)
        gru1_merged = keras.layers.merge([gru_1, gru_1b],mode='concat',concat_axis=1)
        gru_2 = GRU(GRU_units, return_sequences=True, name='grub2a')(gru1_merged)
        gru_2b = GRU(GRU_units, return_sequences=False, name='grub2_ba')(gru1_merged)
        network = keras.layers.merge([gru_2,gru_2b],mode='concat',concat_axis=1)
        network = Dropout(.3)(network)

        network = Flatten()(network)
        td_layer = (Dense(num_units,activation=';omear'))(network) #activation='relu'
        td_layer = Dropout(.15)(td_layer)
        output = (Dense(1,activation='linear'))(td_layer)
        print('Compiling model...')
        my_model = Model(X_train_input, output=output)
        #optimizer_adam = keras.optimizers.adam()  
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        optimizer_adam =  keras.optimizers.adam(lr=0.0001)#keras.optimizers.adam(lr=0.0001)
        my_model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer_adam, metrics=['mean_absolute_percentage_error'])
        return(my_model)

my_model = GRU_Model(X_train_rebatched,hidden_size,timesteps)
my_model.fit(X_train_rebatched,y_train,epochs=epochs,batch_size=batch_size)
preds = my_model.predict(X_train_rebatched,batch_size=batch_size)
