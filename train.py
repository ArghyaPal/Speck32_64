def train_speck_distinguisher(num_blocks, num_filters, num_outputs, d1, d2, 
                                    word_size, ks, final_activation, reg_param,
                                    num_epochs, num_rounds, depth, wdir):  
	#create the network
	model, encoder = make_resnet(num_blocks, num_filters, 
                                      num_outputs, d1, d2, word_size, 
                                      ks, depth, reg_param, 
                                      final_activation, reg_param);
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
