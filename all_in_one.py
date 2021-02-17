from keras_layers import Conv1DTranspose

import numpy as np
import os
from os import urandom
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
#from keras.layers import Conv1DTranspose
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
seed = 123

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

wdir = str(os.getcwd())

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);
  

 # Essential Definitions

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
    return(ks);# Essential Definitions
  
  
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


def make_train_data(n, nr, diff=(0x0040,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y==0);
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);#, plain0l, plain0r, plain1l, plain1r]);
    return(X,Y);
    
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
    return(res);

 
bs = 50; # Batch Size

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
    return(res);
    
def train_speck_distinguisher(num_epochs, num_rounds, depth, num_blocks, num_filters, num_outputs, d1, d2, word_size, ks, reg_param, final_activation, wdir):
    #create the network
    model, encoder = make_resnet(num_blocks, num_filters, num_outputs, d1, d2, word_size, ks, depth, reg_param, final_activation);
    model.compile(optimizer='adam',loss='mse',metrics=['acc']);
    
    #generate training and validation data
    X, Y = make_train_data(10**7,num_rounds);
    X_eval, Y_eval = make_train_data(10**6, num_rounds);
    
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));

    #train and evaluate
    history = model.fit(X,X,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, X_eval), callbacks=[lr,check]);
    np.save(wdir+'history'+str(num_rounds)+'r_depth'+str(depth)+'.npy', history.history['val_acc']);
    np.save(wdir+'history'+str(num_rounds)+'r_depth'+str(depth)+'.npy', history.history['val_loss']);
    dump(history.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(history.history['val_acc']));
    return(model, encoder, history, X, Y);
    
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  
  
  # Encoder
  encoding = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  encoding = BatchNormalization()(encoding);
  encoding = Activation('relu')(encoding);
  shortcut = encoding;
  for i in range(depth):
    encoding = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    encoding = BatchNormalization()(encoding);
    encoding = Activation('relu')(encoding);
    encoding = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(encoding);
    encoding = BatchNormalization()(encoding);
    encoding = Activation('relu')(encoding);
    shortcut = Add()([shortcut, encoding]);


  # Latent Vector
  encoding = Flatten()(shortcut);
  reshape1 = Reshape((16, 32))(encoding)
  
  
  # Decoder
  decoding = Conv1DTranspose(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(reshape1);
  decoding = BatchNormalization()(decoding);
  decoding = Activation('relu')(decoding);
  #dec_shortcut = deconv1;
  #for i in range(depth):
  #  decoding = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(dec_shortcut);
  #  decoding = BatchNormalization()(decoding);
  #  decoding = Activation('relu')(decoding);
  #  decoding = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(decoding);
  #  decoding = BatchNormalization()(decoding);
  #  decoding = Activation('relu')(decoding);
  #  dec_shortcut = Add()([dec_shortcut, decoding]);
  decoding = Conv1DTranspose(num_filters/2, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(decoding);#(dec_shortcut);#
  decoding = BatchNormalization()(decoding);
  decoding = Activation('relu')(decoding);
  decoding = Conv1DTranspose(num_filters/8, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(decoding);
  decoding = BatchNormalization()(decoding);
  decoding = Activation('sigmoid')(decoding);
  out = Flatten()(decoding);
  
  
  autoencoder = Model(inputs=inp, outputs=out);
  encoder = Model(inputs=inp, outputs=encoding);

  #for layer in autoencoder.layers:
  #    print(layer.output_shape)
  #    print("I am Arghya\n")

  return(autoencoder, encoder);
  
bs = 5000;

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def train_speck_distinguisher(num_epochs, num_rounds=5, depth=10):
    
    #create the network
    model, encoder = make_resnet(depth=depth, reg_param=10**-5);
    model.compile(optimizer='adam',loss='mse',metrics=['acc']);
    
    #generate training and validation data
    X, Y = make_train_data(10**7,num_rounds);
    X_eval, Y_eval = make_train_data(10**6, num_rounds);
    
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));

    #train and evaluate
    history = model.fit(X,X,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, X_eval), callbacks=[lr,check]);
    np.save(wdir+'history'+str(num_rounds)+'r_depth'+str(depth)+'.npy', history.history['val_acc']);
    np.save(wdir+'history'+str(num_rounds)+'r_depth'+str(depth)+'.npy', history.history['val_loss']);
    dump(history.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(history.history['val_acc']));
    return(model, encoder, history, X, Y);

    


if __name__ == "__main__":
  model, encoder, history, X, Y = train_speck_distinguisher(num_epochs=200, num_rounds=5, depth=10)
  
  
  
