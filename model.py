# Model
def make_resnet(num_blocks, num_filters, num_outputs, d1, d2, word_size, ks, depth, reg_param, final_activation):
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
	deconv1 = Conv1DTranspose(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(reshape1);
	deconv1 = BatchNormalization()(deconv1);
	deconv1 = Activation('relu')(deconv1);
	deconv2 = Conv1DTranspose(num_filters/2, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(deconv1);
	deconv2 = BatchNormalization()(deconv2);
	deconv2 = Activation('relu')(deconv2);
	deconv3 = Conv1DTranspose(num_filters/8, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(deconv2);
	deconv3 = BatchNormalization()(deconv3);
	deconv3 = Activation('relu')(deconv3);
	out = Flatten()(deconv3);
	# Defining the Authoencoder and Encoder
	autoencoder = Model(inputs=inp, outputs=out);
	encoder = Model(inputs=inp, outputs=encoding);
  
  # Print layers and check dimensions
  #for layer in autoencoder.layers:
  #    print(layer.output_shape)
  #    print("\n")
  
	return(autoencoder, encoder);
#---------------------------------------------#
