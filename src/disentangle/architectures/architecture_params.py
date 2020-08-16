import tensorflow as tf
import copy

default_decoder_activations = {"default":tf.nn.leaky_relu, -1:tf.math.sigmoid}
default_encoder_activations = {"default":tf.nn.leaky_relu, -1:tf.keras.activations.linear}

###############################
# gaussian64 vae architecture #
###############################
encoder64_architecture = [
	[32,4,2],
	[32,4,2],
	[64,4,2],
	[64,4,2],
	[["flatten"],[256]],
	]
decoder64_architecture = [
	[256],
	[[4*4*64],["reshape", [4,4,64]]], 
	[64,4,2],
	[64,4,2],
	[32,4,2],
	[3,4,2],
	]
decoder64_activations = {**default_decoder_activations}
encoder64_activations = {**default_encoder_activations}

################################
# gaussian256 vae architecture #
################################
encoder256_architecture = [
		[[16,5,1], ["ap",2]], 
		[[[32,1,1], [32,3,1]],["ap",2]], # these multiconv groupings are for resnet
		[[[64,1,1], [64,3,1]],["ap",2]], 
		[[[128,1,1], [128,3,1]],["ap",2]], 
		[[[128,3,1], [128,3,1]],["ap",2]], 
		[["flatten"],[1024]],
		[512], # this is the number of latent elements
		]
decoder256_architecture = [
		[512], # this is the number of latent elements
		[1024],
		[[8*8*128], ["reshape", [8,8,128]], ["up",2]], 
		[[[128,3,1], [128,3,1]], ["up",2]], 
		[[[128,1,1], [128,3,1]], ["up",2]], 
		[[[64,1,1], [64,3,1]], ["up",2]], 
		[[[32,1,1], [32,3,1]], ["up",2]], 
		[3,5,1], 
		]
decoder256_activations = {
	**default_decoder_activations, 
	**{i:{"default":default_decoder_activations["default"], 	# these extra activations are for resnet, this is not a generalized method.
		0:{"default":default_decoder_activations["default"],	# projections at the first layer of the block (if resnet uses projection)
			0:tf.keras.activations.linear}} for i in [4,5,6]}}  

encoder256_activations = {
	**default_encoder_activations, 
	**{i:{"default":default_encoder_activations["default"], 	# these extra activations are for resnet, this is not a generalized method.
		0:{"default":default_encoder_activations["default"],	# projections at the first layer of the block (if resnet uses projection)
			0:tf.keras.activations.linear}} for i in [1,2,3]}}  

################################
# gaussian512 vae architecture #
################################
encoder512_architecture = [
		[[16,5,1], ["ap",2]], 
		[[[32,1,1], [32,3,1], [32,3,1]], ["ap",2]],
		[[[64,1,1], [64,3,1], [64,3,1]], ["ap",2]], 
		[[[128,1,1], [128,3,1], [128,3,1]], ["ap",2]], 
		[[[256,1,1], [256,3,1], [256,3,1]], ["ap",2]], 
		[[[256,3,1], [256,3,1]], ["ap",2]], 
		[[[256,3,1], [256,3,1]], ["ap",1]], 
		[["flatten"], [4096]],
		]
decoder512_architecture = [
		[4096], # this is the number of latent elements
		[[8*8*256], ["reshape", [8,8,256]], ["up",1]], 
		[[[256,3,1], [256,3,1]], ["up",2]], 
		[[[256,3,1], [256,3,1]], ["up",2]], 
		[[[256,1,1], [256,3,1], [256,3,1]], ["up",2]], 
		[[[128,1,1], [128,3,1], [128,3,1]], ["up",2]], 
		[[[64,1,1], [64,3,1], [64,3,1]], ["up",2]], 
		[[[32,1,1], [32,3,1], [32,3,1]], ["up",2]], 
		[3,5,1], 
		]
decoder512_activations = {
	**default_decoder_activations, 
	**{i:{"default":default_decoder_activations["default"], 	# these extra activations are for resnet, this is not a generalized method.
		0:{"default":default_decoder_activations["default"],	# projections at the first layer of the block (if resnet uses projection)
			0:tf.keras.activations.linear}} for i in [4,5,6,7]}}

encoder512_activations = {
	**default_encoder_activations, 
	**{i:{"default":default_encoder_activations["default"], 	# these extra activations are for resnet, this is not a generalized method.
		0:{"default":default_encoder_activations["default"],	# projections at the first layer of the block (if resnet uses projection)
			0:tf.keras.activations.linear}} for i in [1,2,3,4]}}



