import numpy as np
import os
from os import urandom
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

# if you are using tensorflow < v2.4.1
from keras_layers import Conv1DTranspose  
# else uncomment the following line
from keras.layers import Conv1DTranspose 

from model import *


#---------------------------------------------#

# Helper Functions
def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);
MASK_VAL = 2 ** WORD_SIZE() - 1;
def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);
def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));
def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);
def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);
def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);
    
#---------------------------------------------#
def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);
def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);
#---------------------------------------------#

def check_testvector():
	key = (0x1918,0x1110,0x0908,0x0100)
	pt = (0x6574, 0x694c)
	ks = expand_key(key, 22)
	ct = encrypt(pt, ks)
	if (ct == (0xa868, 0x42f2)):
		print("Testvector verified.")
		return(True);
	else:
		print("Testvector not verified.")
		return(False);
def convert_to_binary(arr):
	X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
	for i in range(4 * WORD_SIZE()):
		index = i // WORD_SIZE();
		offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
		X[i] = (arr[index] >> offset) & 1;
		X = X.transpose();
	return(X);
#---------------------------------------------# 
  
  
# Dataset Creation
def make_train_data(n, nr, diff=(0x0040,0)):
	Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
	keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  # One bit difference: Palintext-Plaintext pair 
	plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
	plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
	plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  # Randomtext-Plaintext pair
	num_rand_samples = np.sum(Y==0);
	plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
	plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  # Key generation
	ks = expand_key(keys, nr);
  # Ciphertext from palintexts
	ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);   # <---- Use of encryption function on input text to get the Ciphertext
	ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);   # <---- Use of encryption function on input text to get the Ciphertext
  # Dataset 
	X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
	return(X,Y);
#---------------------------------------------#

def make_checkpoint(datei):
	res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
	return(res);
#---------------------------------------------#


def cyclic_lr(num_epochs, high_lr, low_lr):
	res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
	return(res);


if __name__ == "__main__":
# Define all necesary hyperparameters here:
  num_epochs=200
  num_rounds=5
  depth=10
  
  num_blocks=2
  num_filters=32
  num_outputs=1
  
  d1=64
  d2=64
  word_size=16
  ks=3
  reg_param=0.0001
  final_activation='sigmoid  
  
  wdir= str(os.getcwd())+'/model/'
  # Welcome Message
  print("Welcome! You are training Speck autoencoder with encoder resnet depth %d, number of epochs %d, number of rounds %d" % depth, num_epoches, num_rounds)
  print("Files are saved in %s location" %wdir)

	model, encoder, history, X, Y = train_speck_distinguisher(num_blocks, num_filters, num_outputs, d1, d2, 
                                                                word_size, ks, final_activation, reg_param, 
                                                                num_epochs, num_rounds, depth, wdir)
