"""
This module contains functions used to access various datasets
"""
#python packages
import numpy as np
import os
import gzip
import pickle
#import scipy as scp
import random
#import time
#import lycon
import cv2
import h5py

#utils packages
from disentangle.general.tools import * # need the from if importing this file

########################
# Get Various Datasets #
########################
def get_mnist_data(datapath, shuffle=False):
	"""
	Args:
		datapath
			- this is the path for the MNIST dataset.
	Returns:
		training_data
			- training data
			- in the form of a dictionary. "labels": labels, "data":data
		validation_data
			- validation data
			- in the form of a dictionary. "labels": labels, "data":data
		test_data
			- test data
			- in the form of a dictionary. "labels": labels, "data":data
	"""
	data_file = "MNIST/mnist.pkl.gz"
	mnist_path = os.path.join(datapath, data_file)
	with gzip.open(mnist_path, "rb") as f:
		datasets = pickle.load(f, encoding="latin1")
	
	def create_dataset_dict(*args):
		#takes in tuple of data, labels and creates a dictionary.
		new_set = {}
		for i in range(len(args)):
			dataset = args[i]
			data = dataset[0].reshape(-1,28,28, 1)
			if shuffle:
				np.random.shuffle(data)
			labels = dataset[1]
			if new_set == {}:
				new_set["data"] = data
				new_set["labels"] = labels
				new_set["labels_names"] = ["one","two","three","four","five","six","seven","eight","nine","ten"]
			new_set["data"] = np.concatenate((new_set["data"] , data), axis=0)
			new_set["labels"] = np.concatenate((new_set["labels"] , labels), axis=0)
		return new_set
	ret = create_dataset_dict(*datasets)
	return ret["data"], ret["labels"]

def get_celeba_data(datapath, save_new=False, get_group=True, group_num=1, shuffle=False, max_len_only=True, is_HD=256, group_size_if_save=3000,**kwargs):
	"""
	This will retrieve the celeba/celeba-hq datasets. The hdf5, if availabe, should be in the form of:
	- datapath/celeba-hq/celeba-64.hdf5
	- datapath/celeba-hq/celeba-128.hdf5
	- datapath/celeba-hq/celeba-256.hdf5
	- datapath/celeba-hq/celeba-512.hdf5
	- datapath/celeba-hq/celeba-1024.hdf5
	- datapath/celeba/images.hdf5

	If the hdf5 file is not available, the images must be stored as:
	- datapath/celeba-hq/celeba-64/
	- datapath/celeba-hq/celeba-128/
	- datapath/celeba-hq/celeba-256/
	- datapath/celeba-hq/celeba-512/
	- datapath/celeba-hq/celeba-1024/
	- datapath/celeba/images/
	The labels must also be stored, especially the list_attr_celeba.txt file within
	- datapath/celeba-hq/
	- datapath/celeba/
	

	Examples:
		>>> dataset, get_group = gu.get_celeba_data(gc.datapath, group_num=1)
		>>> images_1, labels_1 = get_group()
		>>> images_2, labels_2 = get_group()

	Args:
		datapath:  This is the datapath the the general data folder
		save_new:  saves a hdf5 dataset if true, or not available. uses old one if false
		get_group:  Makes this function return dataset and group objects (for chunked data loading),
			otherwise, load all images and labels. get_group is an iterator which will load data.
		group_num:  The number of groups to load at once
		shuffle:  Shuffles the groups, if loading with groups
		max_len_only:  This will force the groups to be of max length.
		is_HD: Whether to extract the hd version (int - 64, 128, 256, 512, 1024) or default version (False)
		**kwargs: These are any other irrelevant kwargs.

	Returns:
		data, labels if not get group, else dataset object, get_group object.
	"""
	#loads from numpy file, if available or specified, else save to numpy file
	#should make these paths into constants in a python file at the directory, or in general constants
	
	##############################
	# Choose CelebA or CelebA-HQ #
	##############################
	if is_HD:
		datapath = os.path.join(datapath, "celeba-hq")
		images_path = "celeba-%d"%is_HD
	else:
		datapath = os.path.join(datapath, "celeba")
		images_path = "images"
	
	###############################################################################################
	# Getting the paths and files for raw data (incase needs to recreated dataset) and saved data #
	###############################################################################################
	labels_file = "list_attr_celeba.txt"
	images_saved_file = "%s.hdf5"%images_path
	
	images_saved_path = os.path.join(datapath, images_saved_file)
	labels_path = os.path.join(datapath, labels_file)
	images_path = os.path.join(datapath, images_path)

	#########################################################
	# Initialize Dataset Object and Set Save New Parameters #
	#########################################################
	dataset = GetData(images_saved_path)
	if not save_new:
		ret = dataset.possible_load_group_indices(images_saved_path, shuffle)
		if ret:
			save_new = True

	######################
	# Save a New Dataset #
	######################
	if save_new:
		if os.path.exists(images_saved_path):
			os.remove(images_saved_path)
		#get the labels and images filenames:
		#get previous saved path.
		with open(labels_path,"r") as f:
			total_labels = f.readlines()

		#get the labels:
		filenames = []
		labels = []
		labels_names = total_labels[1].split()
		print("loading labels...")
		incongruency_counter = 0
		for line in total_labels[2:]:
			filepath = os.path.join(images_path, line.split()[0])
			if os.path.exists(filepath):
				labels.append(line.split()[1:])
				filenames.append(os.path.join(images_path, line.split()[0]))
			else:
				incongruency_counter+=1
		if incongruency_counter:
			print("Skipped labels, ", incongruency_counter, "remaining: ", len(filenames))
		labels = (np.asarray(labels).astype(int)+1)/2
		
		dataset.save_by_group(labels, filenames, group_size_if_save)
	
	########################
	# Get Dataset Iterator #
	########################
	if get_group:
		#returns dataset_object and the get next method
		dataset.possible_load_group_indices(shuffle, max_len_only)
		return dataset, _Dataset(dataset=dataset, group_num=group_num)
	
	##############################
	# Directly load the new data #
	##############################
	else:
		dataset.load()

	return dataset.images, dataset.labels


###################
# Dataset Objects #
###################
class _Dataset():
	def __init__(self, dataset, group_num=1, random_selection=True, remove_past=False):
		self.dataset = lambda group_num=group_num, random_selection=random_selection, remove_past=remove_past: dataset.get_next_group(
			group_num=group_num, random_selection=random_selection, remove_past_groups=remove_past)

	def __call__(self, group_num=1, random_selection=True, remove_past=False):
		return self.dataset(group_num=group_num, random_selection=random_selection, remove_past=remove_past)

class GetData():
	#this will get the image data.
	def __init__(self, save_path):
		self.images = None
		self.labels = None
		self.groups_list = None
		self.cur_group_index = 0
		self.max_len = None
		self.last_group_list = None
		self.data_savepath = save_path

	def get_group_size(self):
		save_path = self.data_savepath
		with h5py.File(save_path, "r") as file:
			group_size = file["images"]["0"].shape[0]
		return group_size

	def load(self, group_indices=None):
		"""
		This will load the groups, given the indices
		:param group_indices: the indicies of the group to load
		:return: 0: success, -1: no path found
		"""
		if not os.path.exists(self.data_savepath):
			print("Must call possible_load_group_indices first!")
			return -1
		#load the data
		total_images = None
		total_labels = None
		if group_indices is None: 
			group_indices = self.groups_list

		#timer = Timer()
		with h5py.File(self.data_savepath, "r") as file:
			for i,v in enumerate(group_indices, 0):
				current_images = file["images"][v][()]
				current_labels = file["labels"][v][()]
				#""" This fills a premade array, this is faster than just concatenation
				group_size = current_images.shape[0]
				if total_images is None:
					total_images = np.empty((len(group_indices)*group_size, *current_images.shape[1:]), dtype=np.uint8)
					total_labels = np.empty((len(group_indices)*group_size, *current_labels.shape[1:]))
				#timer.print("Time to load values: ")
				total_images[i*group_size:(i+1)*group_size] = current_images
				total_labels[i*group_size:(i+1)*group_size] = current_labels
				#"""
				# below adds to list then vstacks, which is slightly slower than above after testing.
				"""
				if total_images is None:
					total_images = []
					total_labels = []
				total_images.append(current_images)
				total_labels.append(current_labels)
				timer.print("TEST %d"%i)
			total_images = np.vstack(total_images)
			total_labels = np.vstack(total_labels)
			timer.print("TEST")
			#"""
		self.images = total_images
		self.labels = total_labels
		return 0

	def batch(self, batch_size, group_num=1, random_selection=True, remove_past=False):
		dataset_call = lambda group_num=group_num, random_selection=random_selection, remove_past=remove_past: self.get_next_group(
			group_num=group_num, random_selection=random_selection, remove_past_groups=remove_past)
		return DatasetBatch(dataset_call, batch_size)

	def get_next_group(self, random_selection=True, group_num=1, remove_past_groups=False):
		"""
		This function is an iterator, will iterate through the groups in an hdf5 file.
		:param random_selection: whether to select the group randomly or not.
		:param group_num: The number of groups to load per batch.
		:param remove_past_groups: This is a boolean, if true, will remove the next group number(s) from the iterating
		dataset, otherwise, iterate in a loop, as each get_next_group() is called.
		:return:images, labels.
		"""
		#gets the next group, either a random selection, or increment the list
		assert group_num > 0
		groups=[]
		assert len(self.groups_list), "no more groups, empty groups array."
		for i in range(group_num):
			idx = self.cur_group_index if not random_selection else random.randint(0, len(self.groups_list)-1)
			if not remove_past_groups:
				groups.append(self.groups_list[idx % len(self.groups_list)])
				self.cur_group_index+=1
			else:
				groups.append(self.groups_list.pop(idx % len(self.groups_list)))
		self.load(groups)
		self.last_group_list = groups
		return self.images, self.labels

	def possible_load_group_indices(self, shuffle=True, max_len_only=False):
		"""
		This is the possible indices that you can pick a group from.
		:param shuffle: group_indices the indicies of the group to load
		:param max_len_only: This will force the groups to be of max length.
		:return: groups_list: possible groups to load, -1: no path found
		"""
		#gets the possible groups to load.
		save_path = self.data_savepath
		if not os.path.exists(save_path):
			print("no path found!")
			return -1

		#load the data
		with h5py.File(save_path, "r") as file:
			groups_list= [k for k in file["images"].keys()]
			max_len = max([file["images"][k].attrs["length"] for k in groups_list])
			groups_list = [k for k in file["images"].keys() if not max_len_only or file["images"][k].attrs["length"] >= max_len]
			groups_list = [i for _,i in sorted(zip([int(i) for i in groups_list], groups_list))]
			if shuffle:
				random.shuffle(groups_list)
		self.groups_list = groups_list
		return 0


	def save_file(self, groups=1000, dataset_offset=0):
		"""
		loads images from a saved path and sets it as self.images
		Current assumptions:
			- the data and labels are of the same size in the 0th axis
			- the corresponding data and labels are a 1 to 1 mapping. 
		Args:
			groups
				- n sized groups to split the data into.
				- default is 1000 datapoints/group
			dataset_offset
				- This is number to be added to i when saving dataset numbers, only affects the name.
		Returns:
			 0: success
			-1: no data found
		"""
		save_path = self.data_savepath
		if self.images is None and self.labels is None:
			print("no data to save!")
			return -1
		elif self.images is None or self.labels is None:
			item_unavailable = "images" if self.images is None else self.labels
			print("Warning! %s not loaded!"%item_unavailable)
		else:
			pass
		assert len(self.labels) == len(self.images)

		with h5py.File(save_path, "a") as file:
			if not "labels" in file:
				labels_grp = file.create_group("labels")
			else: 
				labels_grp = file["labels"]
			if not "images" in file:
				images_grp = file.create_group("images")
			else: 
				images_grp = file["images"]
			num_data = self.labels.shape[0]
			for i in range(num_data//groups+int(bool(num_data%groups))):
				data_start_index = i*groups
				data_end_index = data_start_index+min(groups, num_data-data_start_index)
				#print("%d"%(i+dataset_offset))
				ldset = labels_grp.create_dataset("%d"%(i+dataset_offset), data=self.labels[data_start_index:data_end_index])
				idset = images_grp.create_dataset("%d"%(i+dataset_offset), data=self.images[data_start_index:data_end_index])
				ldset.attrs["length"] = data_end_index - data_start_index
				idset.attrs["length"] = data_end_index - data_start_index

	def save_by_group(self, labels, filenames, groups):
		"""
		loads in data by groups and saves them accordingly.

		groups is the max amount of images per group
		"""
		save_path = self.data_savepath
		num_data = len(filenames)
		num_groups = num_data//groups+int(bool(num_data%groups))
		print("num images per group: ", int(num_data/num_groups), "total: ", num_data)
		for i in range(num_groups):
			print("\rgroups loaded: "+loading_bar(i, num_groups), end="")
			start_num = i*groups
			self.set_labels(labels[start_num:(i+1)*groups])
			self.get_images_from_filenames(filenames[start_num:(i+1)*groups], False)
			self.save_file(groups, dataset_offset=start_num)

	def set_labels(self, labels):
		#sets the labels
		self.labels = np.asarray(labels)


	def get_images_from_filenames(self, filenames_list, print_loading_bar=True):
		#given a list of filenames, will retrieve the image data
		"""
		Current assumptions:
			- images are of the same size
		
		Args:
			filenames_list:
				- list of the filenames for each of the images.
		"""
		#tf.enable_eager_execution()
		images = None
		#filenames_list = np.asarray(filenames_list).reshape(-1,)
		for i in range(len(filenames_list)):
			if print_loading_bar:
				print("\r"+loading_bar(i, len(filenames_list)), end="")
			#image = lycon.load(filenames_list[i])
			image = cv2.imread(filenames_list[i])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			#print("MIN, MAX", np.amin(image.numpy()), np.amax(image.numpy()))
			if images is None:
				images = np.zeros((len(filenames_list), *image.shape), np.uint8)
				images[i] = image
			else:
				images[i] = image
		if print_loading_bar:
			print()
		self.images = images
		#tf.disable_eager_execution()

class DatasetBatch():

	"""This is a class that goes hand in hand with the GetData class
	This will batch and reload groups accordingly
	"""

	def __init__(self, dataset, batch_size, prep_func=None):
		"""Initialize
		
		Args:
		    dataset (function): This is a function, when called, will produce inputs and outputs.
		    batch_size (int): This is the size we want to batch the data into
		    prep_func (function, optional): Additional preprocessing batch to be returned, this will be saved, and should be used for recurring, computationally heavy processing functions
		"""
		self.batch_size = batch_size
		self.dataset = dataset
		self.i = 0 # current iteration on current dataset sample
		self.inputs, self.outputs = self.dataset()

	def get_next(self):
		"""This selection method will sample from the batch
			and get a new batch every dataset/batchsize 
			returns [batch_size]
		"""
		dset_shape = self.inputs.shape[0]
		max_iteration = dset_shape//self.batch_size
		self.i+=1

		indices_to_take = np.random.randint(dset_shape, size=(self.batch_size))

		batched = (np.take(self.inputs, indices_to_take, axis=0), np.take(self.outputs, indices_to_take, axis=0))

		if self.i >= max_iteration:
			self.inputs, self.outputs = self.dataset()
			self.i = 0

		return batched

	def __call__(self):
		return self.get_next()

	def __iter__(self):
		return self

	def __next__(self):
		return self.__call__()

