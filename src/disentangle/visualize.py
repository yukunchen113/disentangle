import numpy as np
import tensorflow as tf
from disentangle.general import Timer, loading_bar
import cv2
import os
import imageio
import shutil
class Traversal:
	
	def __init__(self, model, inputs):
		"""Traversal of the latent space class give one model and one set of inputs
		
		Examples:
			>>> def image_traversal(model, inputs, latent_of_focus=3, min_value=0, max_value=3, num_steps=15, is_visualizable=True)
			>>> 	traverse = Traversal(model, inputs)
			>>> 	traverse.traverse_latent_space(latent_of_focus=3, min_value=0, max_value=3, num_steps=15)
			>>> 	traverse.create_samples()
			>>> 	if not is_visualizable:
			>>> 		return traverse.samples
			>>> 	return traverse.construct_single_image()


		Args:
		    model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
		    inputs (numpy arr): Input images in NHWC
		    latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
		    min_value (int): min value for traversal
		    max_value (int): max value for traversal
		    num_steps (int): The number of steps between min and max value
		
		Returns:
		    Numpy arr: image
		"""
		self.model = model
		self.orig_inputs = inputs
		self.inputs = inputs
		self.samples = None
		self.latent_rep_trav = None #latent traversal to become shape [num traversal, N, num latents]

	@property
	def num_latents(self):
		return self.model.num_latents

	def traverse_complete_latent_space(self, min_value=-3, max_value=3, num_steps=30):
		"""Will travers all the latent space. 
		The num images and num latents dimensions will be flattened to one dimension
		shape of latents will be: [num images, num latents]
		
		Args:
		    min_value (int): min value for traversal
		    max_value (int): max value for traversal
		    num_steps (int): The number of steps between min and max value
		
		"""
		latent_reps = []
		inputs = None
		
		# accumulate images for all the different latent representations, for all images
		for i in range(self.num_latents):
			self.traverse_latent_space(latent_of_focus=i, 
				min_value=min_value, max_value=max_value, num_steps=num_steps)
			latent_reps.append(self.latent_rep_trav.copy())
			
			if inputs is None:
				inputs = np.empty((self.num_latents, *self.inputs.shape))
			inputs[i] = self.inputs

		# latents
		latent_reps = np.asarray(latent_reps)
		latent_reps = np.transpose(latent_reps, (2,0,1,3)) 
		self.latent_rep_trav = latent_reps.reshape((-1, *latent_reps.shape[-2:])).transpose((1,0,2))
		# self.latent_rep_trav is flattened from [num_images, num_latents]



		# inputs duplication
		inputs = np.transpose(inputs, (1,0,2,3,4))
		inputs = np.reshape(inputs, (-1, *inputs.shape[-3:]))
		self.inputs = inputs

	def encode(self, inputs):
		return self.model.encoder(inputs)

	def decode(self, samples):
		return self.model.decoder(samples)

	def traverse_latent_space(self, latent_of_focus=3, min_value=-3, max_value=3, num_steps=30, add_min_max=False):
		"""traverses the latent space, focuses on one latent for each given image.
		
		Args:
		    latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
		    min_value (int): min value for traversal
		    max_value (int): max value for traversal
		    num_steps (int): The number of steps between min and max value
		
		"""
		t = Timer()
		# initialize latent representation of images
		_, latent_rep, latent_logvar = self.encode(self.inputs)
		latent_rep = latent_rep.numpy()
		stddev = np.sqrt(np.exp(latent_logvar.numpy()[:,latent_of_focus]))

		# create latent traversal
		latent_rep_trav = []
		for i in np.linspace(min_value, max_value, num_steps):
			mod_latent_rep = latent_rep.copy()
			addition = np.zeros(mod_latent_rep.shape)
			addition[:,latent_of_focus] = i
			mod_latent_rep=latent_rep
			if not add_min_max:
				mod_latent_rep[:,latent_of_focus]=0
			mod_latent_rep+=addition
			latent_rep_trav.append(mod_latent_rep.copy())


		self.latent_rep_trav = np.asarray(latent_rep_trav)
		self.inputs = self.orig_inputs


	def create_samples(self, batch_size=16):
		"""Creates the sample from the latent representation traversal
		"""
		assert not self.latent_rep_trav is None, "Please call traverse_latent_space first to get latent elements for self.latent_rep_trav"

		# flattened latent traversal for one batch dimension (assuming that the latent traversal is of the size, [num traversal, N, num latents])
		latent_rep = np.vstack(self.latent_rep_trav)

		# get the samples
		generated = None
		for i in range(np.ceil(latent_rep.shape[0]/batch_size).astype(int)):
			gen = self.decode(latent_rep[i*batch_size:(i+1)*batch_size]).numpy()
			if generated is None:
				generated = np.empty((latent_rep.shape[0],*gen.shape[1:]))
			generated[i*batch_size:(i+1)*batch_size] = gen
		# reshape back to [num traversal, N, W, H, C], as per self.latent_rep_trav
		self.samples = tf.reshape(generated, (*self.latent_rep_trav.shape[:2],*generated.shape[1:])).numpy()
	
	def construct_single_image(self):
		"""Contruct a single image to be displayed from samples. samples should be of shape [num traversal, N, W, H, C]
		
		Returns:
		    numpy array: array of images
		"""
		assert not self.latent_rep_trav is None, "Please call create_samples first to get sample to reconstruct"

		samples = np.concatenate(self.samples,-2) # concatenate horizontally
		samples = np.concatenate(samples,-3) # concatenate vertically
		real = np.concatenate(self.inputs,-3)

		image = np.concatenate((real, samples),-2) #concatenate the real and reconstructed images
		image = image[:,:,:3]
		return image
	@property
	def samples_list(self):
		inputs = np.broadcast_to(self.inputs, self.samples_list[0].shape)
		return [inputs, self.samples]

	def save_gif(self, gif_path, latent_num=None):
		"""Save traveral as a gif
		
		Args:
		    gif_path (nparray): the destination of the gif. Must end with .gif
		    latent_num (int, optional): the latent number to animate. If this is None, will animate entire latent space 
		"""
		# vertically stack all sample list
		# include inputs
		
		samples = np.concatenate(self.samples_list, -3)

		# horizontally stack images
		samples = samples.transpose(1,0,2,3,4)
		if latent_num is None:
			samples = np.concatenate(samples, axis=-2)	
		else:
			samples = samples.reshape(-1, self.num_latents, *samples.shape[1:])
			samples = np.concatenate(samples[:,latent_num], axis=-2)

		create_gif(samples, gif_path)

def create_gif(arr, gif_path):
	"""Creates samples from a NHWC data array and saves it into gif_path.
	arr must be a float between 0 and 1
	
	Args:
	    arr (nparray): gif array
	    gif_path (str): path to save gif (must end with .gif)
	"""
	tmp_dir = os.path.splitext(gif_path)[0]+"_tmp_gif"
	if os.path.exists(tmp_dir):
		# remove previous exising temp
		shutil.rmtree(tmp_dir)
	os.mkdir(tmp_dir)
	frames_path = os.path.join(tmp_dir,"{i}.jpg")
	arr = np.concatenate((arr, np.flip(arr, axis=0)), 0)


	num_images = len(arr)
	for i, x in enumerate(arr):
		rgb_img = (x[:,:,:3]*255).astype(np.uint8)

		cv2.imwrite(frames_path.format(i=i), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY),100])
		print("Creating Gif Frames: ", loading_bar(i+1, len(arr)), end="\r")
	
	with imageio.get_writer(gif_path, mode='I') as writer:
		for i in range(num_images):
			writer.append_data(imageio.imread(frames_path.format(i=i)))
			print("Stitching Gif Frames: ", loading_bar(i+1, len(arr)), end="\r")
	shutil.rmtree(tmp_dir)
	print("\nCreated gif: %s"%gif_path)
