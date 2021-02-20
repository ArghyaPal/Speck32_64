import numpy as np
import os
from os import urandom
from pickle import dump

import math
from sklearn.manifold import TSNE
import matplotlib.colors as c
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from subprocess import call

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

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
  X = np.zeros((2 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(2 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);
  
 
# Make train dataset
 
def make_train_data(n, nr, diff=(0x0040,0)):
	Y[:int(n/2)] = 1;
	Y[int(n/2):] = 0;
	Y = Y & 1;
	keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
	plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
	plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
	ks = expand_key(keys, nr);
	ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
	ctdata0l[Y==0] = plain0l[Y==0]
	ctdata0r[Y==0] = plain0r[Y==0]
	X = convert_to_binary([ctdata0l, ctdata0r]);
	return(X, Y);

# Make classifier
  
def make_resnet(num_blocks, num_filters, num_outputs, 
		d1, d2, word_size, ks, depth, reg_param, 
		final_activation):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size,));#* 2,));
  rs = Reshape((num_blocks, word_size))(inp);#2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  encoding = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  encoding = BatchNormalization()(encoding);
  encoding = Activation('relu')(encoding);
  #add residual blocks
  shortcut = encoding;
  for i in range(depth):
    encoding = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    encoding = BatchNormalization()(encoding);
    encoding = Activation('relu')(encoding);
    encoding = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(encoding);
    encoding = BatchNormalization()(encoding);
    encoding = Activation('relu')(encoding);
    shortcut = Add()([shortcut, encoding]);
  #add prediction head
  flat = Flatten()(shortcut);
  # Dense layers
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  
  # Creating models
  model = Model(inputs=inp, outputs=out);
  encoder = Model(inputs=inp, outputs=encoding);
  
  return(model, encoder);
  

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def train_speck_distinguisher(num_epochs, num_rounds, depth, num_blocks, 
			      num_filters, num_outputs, d1, d2, 
			      word_size, ks, reg_param, final_activation,
			     wdir, bs, num_train_data, num_val_data):
    #create the network
    net, encoder = make_resnet(num_blocks, num_filters, num_outputs, d1, 
			       d2, word_size, ks, depth, reg_param, 
			       final_activation);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    #generate training and validation data
    X, Y = make_train_data(num_train_data,num_rounds);
    X_eval, Y_eval = make_train_data(num_val_data, num_rounds);
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, encoder, h, X, Y);

    
def vis_data(x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim):
    cmap = plt.get_cmap('rainbow', 10)
    # 3-dim vis: show one view, then compile animated .gif of many angled views
    if vis_dim == 3:
        # Simple static figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                c=y_train[:n_predict], cmap=cmap, edgecolor='black')
        fig.colorbar(p, drawedges=True)
        plt.show()
        # Build animation from many static figures
        if build_anim:
            angles = np.linspace(180, 360, 20)
            i = 0
            for angle in angles:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.view_init(10, angle)
                p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                        c=y_train[:n_predict], cmap=cmap, edgecolor='black')
                fig.colorbar(p, drawedges=True)
                outfile = 'anim/3dplot_step_' + chr(i + 97) + '.png'
                plt.savefig(outfile, dpi=96)
                i += 1
            call(['convert', '-delay', '50', 'anim/3dplot*', 'anim/3dplot_anim_' + str(n_train) + '.gif'])
    # 2-dim vis: plot and colorbar.
    elif vis_dim == 2:
        plt.scatter(x_train_encoded[:,0], x_train_encoded[:,1], 
                c=y_train[:n_predict], edgecolor='black', cmap=cmap)
        plt.colorbar(drawedges=True)
        plt.show() 


# Main function
if __name__ == "__main__":

	# All training variables
	num_epochs = 5
	batch_size = 5
	num_rounds = 8
	depth = 10
	num_train_data = 10**7
	num_val_data = 10**6
	
	num_blocks=2
	num_filters=32
	num_outputs=1
	d1=64
	d2=64
	word_size=16
	ks=3
	reg_param=0.0001
	final_activation='sigmoid'
	
	# t-SNE variables
	train_new = False
	n_train = 1000
	predict_new = False
	n_predict = 10000
	vis_dim = 2
	build_anim = False
	
	# Welcome message
	print("Training vanilla speck for Round %d" % num_rounds)
	
	# Dataset storing
	wdir = os.getcwd() + '/model/'
	print(wdir)
        
        # Output from the network
	net, encoder, h, X, Y = train_speck_distinguisher(num_epochs, num_rounds, depth, num_blocks, 
							  num_filters, num_outputs, d1, d2, 
							  word_size, ks, reg_param, final_activation,
							 wdir, batch_size, num_train_data, num_val_data);
	encoder.save_weights(wdir + "encoder_save_weight.h5")
	net.save_weights(wdir + "net_save_weight.h5")
	
	encoder.save(wdir + "encoder_save.h5")
	net.save(wdir + "net_save.h5")
	
	# Perform t-SNE
	#x_train_predict = encoder.predict(X[:n_predict])
	#print("Performing t-SNE dimensionality reduction of Round %d..."% num_rounds)
	#x_train_encoded = TSNE(n_components=3).fit_transform(x_train_predict)
	#print("Done.")

	# Visualize result.
	#vis_data(x_train_encoded, Y, vis_dim, n_predict, n_train, build_anim=True)
