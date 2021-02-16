import numpy as np
import os
from os import urandom
from pickle import dump


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
