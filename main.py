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

quandl_tkn = "" #quandl_token
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
    base = 'http://www.google.com/finance/historical?q=GOOGLEINDEX_US:'
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



#Start of tf model
hidden_size = 128 #LSTM
n_units = 1024 #Dense
batch_size = 5
timesteps = 1
num_input = X_train_normalized.shape[1]
num_batches = int(round(len(my_df)/batch_size,0))
epochs = 5
num_layers = 3

with tf.Graph().as_default():    
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    x = tf.placeholder(tf.float32, [batch_size,timesteps, num_input],name='x')
    y = tf.placeholder(tf.float32, [batch_size],name='y')
        
    cell_state = tf.placeholder(tf.float32, [batch_size, hidden_size],name='c')  #nonlienarity from i and forget gate and previous c
    hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_size],name='h') #nonlinearity from cell_state & outputgate
    init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state) #internal state of LSTM   
                                              
    #Weights that go into Dense Layer after LSTM
    W_into_dense = tf.Variable(tf.truncated_normal(shape=[hidden_size,n_units],stddev=.001,dtype=tf.float32))
    b_into_dense = tf.Variable(tf.zeros(shape=[n_units]), dtype=tf.float32)
    
    #Weights for final output
    W_out = tf.Variable(tf.truncated_normal(shape=[n_units,1]))
    b_out = tf.Variable(tf.zeros(shape=[batch_size,1]))
    
    
    # Forward passes
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    output_lstm,state = tf.nn.dynamic_rnn(cell,x,initial_state=init_state)
    
#
    
    output_lstm = tf.reshape(output_lstm,shape=[batch_size,hidden_size]) #unsure about this
    
    #Into Deep Layer
    dense = tf.nn.relu(tf.matmul(output_lstm,W_into_dense)+b_into_dense) #RELU
    output = tf.matmul(dense,W_out)+b_out #Last Layer is Linear
    output = tf.reshape(output,shape=[batch_size])
#    print(output)
    
    losses = tf.losses.mean_squared_error(labels=y,predictions=output)
    cost = tf.reduce_mean(losses)
    
    opt = tf.train.AdamOptimizer().minimize(cost)


    
    
#    output,state = rnn_model('lstm1',n_hidden,x)
#    
#    preact_for_dense = output_activations(output[-1],W1,b1)
#    dense = tf.nn.relu(preact_for_dense)    
#    pred = tf.matmul(dense,W2)+b2 #i.e. a linear output
#     
#    losses = tf.losses.mean_squared_error(labels=y,predictions=pred)
#    cost = tf.reduce_mean(losses)
#    opt = tf.train.AdamOptimizer().minimize(cost)



    sess = tf.InteractiveSession()
#    with sess:
    
    #Training
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        _current_cell_state = np.zeros((batch_size, hidden_size)).astype(np.float32)+.001
        _current_hidden_state = np.zeros((batch_size, hidden_size)).astype(np.float32)+.001
        for batch in range(num_batches):
            batch_x = np.array(X_train_normalized.ix[batch:batch+batch_size,:])
            batch_y = np.array(spy_vol.ix[batch:batch+batch_size])   
#                batch_y_reshape = batch_y.reshape(batch_size,timesteps)
            batch_x_reshape = batch_x.reshape(batch_size,timesteps,num_input)
            #batch_x_unpacked = tf.unstack(batch_x_reshape,axis=1) 

            _opt ,_cost, _current_state, _output = sess.run(
            [opt,cost, state, output],
            feed_dict={
                x: batch_x_reshape,
                y: batch_y,
                cell_state: _current_cell_state,
                hidden_state: _current_hidden_state

            })
            _current_cell_state, _current_hidden_state = _current_state
        print ('Epoch {}: {}'.format(epoch,_cost))
        print ('Epoch {}:'.format(_output))             

#    #Test
#    print ("Test Preds")
#    for batch in range(num_batches):        
#        x_test = np.array(X_test_normalized.ix[batch:batch+batch_size])
#        x_test_reshape = x_test.reshape(batch_size,timesteps,num_input) #reshape for 
#        _output = sess.run([output],feed_dict={x:x_test_reshape,
#                                    cell_state:_current_cell_state,
#                                    hidden_state:_current_hidden_state})
#        print ('Outputs {}'.format(_output))
#        print ('Y {}'.format(y_test[batch:batch_size+batch]))

    
        
            
            
        
