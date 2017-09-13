#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:03 2017

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
test_batch_size = 32
timesteps = 10
num_input = X_train_normalized.shape[1]
epochs = 20
num_layers = 10

#Reformat the data to have a lookback of 10, which means creating a new dataframe of size
X_train_rebatched = series_to_supervised(X_train_normalized,n_in=timesteps-1)
y_train = y_train[X_train_rebatched.index].astype(np.float32)

X_test_rebatched = series_to_supervised(X_test_normalized,n_in=timesteps-1)
y_test = y_test[X_test_rebatched.index].astype(np.float32)

X_train_rebatched = np.array(X_train_rebatched).reshape(-1,timesteps,num_input).astype(np.float32)
X_test_rebatched = np.array(X_test_rebatched).reshape(-1,timesteps,num_input).astype(np.float32)
num_batches = int(np.round(len(X_train_rebatched)/batch_size))


def Model(x,batch_size,hidden_size,n_units):
    #Weights that go into Dense Layer after GRU
    W_into_dense = tf.Variable(tf.truncated_normal(shape=[hidden_size,n_units],stddev=.001,dtype=tf.float32))
    b_into_dense = tf.Variable(tf.zeros(shape=[n_units]), dtype=tf.float32)
    
    #Weights for final output
    W_out = tf.Variable(tf.truncated_normal(shape=[n_units,batch_size]))
    b_out = tf.Variable(tf.zeros(shape=[batch_size]),validate_shape=True)
    
    #GRU cells

    GRU_through_time = tf.contrib.rnn.MultiRNNCell([my_GRU() for _ in range(num_layers)])
    output, state = tf.nn.dynamic_rnn(GRU_through_time,inputs=X,dtype=tf.float32)
#    output = tf.reshape(output,shape=[batch_size,hidden_size])#unsure about this
    
    output = output[:,-1,:]

    dense = tf.nn.tanh(tf.matmul(output,W_into_dense)+b_into_dense)
    output2 = tf.matmul(dense,W_out)+b_out

#    output2 = tf.reshape(output2,shape=[batch_size])
    #+b_out
    
    return(output2)





with tf.Graph().as_default():    

    
    sess = tf.Session()

    with sess:
            
        dataset_X = tf.contrib.data.Dataset.from_tensor_slices(X_train_rebatched)
        dataset_X = dataset_X.batch(batch_size)
        iterator_X = dataset_X.make_initializable_iterator()
        
        dataset_Y = tf.contrib.data.Dataset.from_tensor_slices(y_train)
        dataset_Y = dataset_Y.batch(batch_size)
        iterator_Y = dataset_Y.make_initializable_iterator()
        
        
        test_dataset_X = tf.contrib.data.Dataset.from_tensor_slices(X_test_rebatched)
        test_dataset_Y = tf.contrib.data.Dataset.from_tensor_slices(y_test)
        
        test_dataset_X = test_dataset_X.batch(test_batch_size)
        iterator_test_X = test_dataset_X.make_initializable_iterator()
        
        test_dataset_Y = test_dataset_Y.batch(test_batch_size)
        iterator_test_Y = test_dataset_Y.make_initializable_iterator()
        
    
        # X, input shape: (batch_size, time_step_size, input_vec_size)
        x = tf.placeholder(tf.float32, [None,timesteps, num_input],name='x')
        y = tf.placeholder(tf.float32, [None],name='y')
        
        #Test
#        x_test_ = tf.placeholder(tf.float32,[None,timesteps,num_input],name='x_test')
#        y_test_ = tf.placeholder(tf.float32,[None],name='y')
        
        sess.run(iterator_X.initializer, feed_dict={x: X_train_rebatched})
        sess.run(iterator_Y.initializer, feed_dict = {y:y_train})
        
#        sess.run(iterator_test.initializer, feed_dict={x: X_test_rebatched,
#                                                 y: y_test})
        
        X = iterator_X.get_next()
        Y = iterator_Y.get_next()
        
        y_temp = Y
        y_temp = tf.reshape(y_temp,shape=[batch_size])
        
        train_output = Model(X,batch_size,hidden_size,n_units) #Training
        
#        with tf.variable_scope('GRU_test'):
#            test_output = Model(X,batch_size,hidden_size,n_units) #Testing
    
    
        losses = k.mean_squared_error(y_true=y_temp,y_pred=train_output)
        cost = tf.reduce_mean(losses)
        
        
        opt = tf.train.AdamOptimizer().minimize(cost)
    
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        
        
        # Training stage
        for epoch in range(epochs):
            sess.run([iterator_X.initializer,iterator_Y.initializer],
                     feed_dict={x: X_train_rebatched,
                                y: y_train})
            for batch in range(num_batches):    
                try:
                # Run optimization
                    _opt,_outputs = sess.run([opt,train_output])
                # Reload the iterator when it reaches the end of the dataset
                except tf.errors.OutOfRangeError:
                    sess.run([iterator_X.initializer,iterator_Y.initializer],
                         feed_dict={x: X_train_rebatched,
                                    y: y_train})
                    _opt,_outputs = sess.run([opt,train_output])
#                    print ('batch {}: Output: {}'.format(batch,_outputs)) 

           #Testing Stage
            sess.run([iterator_test_X.initializer,iterator_test_Y.initializer],feed_dict={x: X_test_rebatched,y: y_test})
            _t_output = sess.run(train_output)
            print ('Test {}: {}'.format(epoch,_t_output))
       
#TODO FIX SHAPES    
                

        
        
                    
      
                
                    
                    
    
           
        
    #    for epoch in range(epochs):
    #        _current_cell_state = np.zeros((batch_size, hidden_size)).astype(np.float32)+.001
    #        _current_hidden_state = np.zeros((batch_size, hidden_size)).astype(np.float32)+.001

#        next_element = iterator.get_next()
#
#        for batch in range(num_batches):
#            batched_data = sess.run(next_element)        
#            batch_x = batched_data[0]; batch_y = batched_data[1]
#
#            
##           batch_y = np.array(y_train.ix[batch:batch+batch_size])   
##                batch_y_reshape = batch_y.reshape(batch_size,timesteps)
##            batch_x_reshape = batch_x.reshape(batch_size,timesteps,num_input)
#            #batch_x_unpacked = tf.unstack(batch_x_reshape,axis=1) 
#
#            _opt ,_cost,_output2= sess.run(
#            [opt,cost,output],feed_dict={batch_x,batch_y})
#
#            print ('Epoch {}: {}'.format(epoch,_output2))