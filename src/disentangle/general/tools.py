"""
This module contains the general tools used, can apply to anything python related

"""
import os
import glob
import configparser as cp
import shutil
import time
def loading_bar(cur, total):
	fraction = cur/total
	string = "[%-20s]\t%.2f%%\t%d/%d\t\t"%("="*int(20*fraction), fraction*100, cur, total)
	return string

def find_largest_factors(num):
	assert num == num//1
	possible_factors = list(range(int(num//num**0.5+1)))
	possible_factors.reverse()
	for i in possible_factors:
		if num/i == num//i:
			return int(i), int(num/i)

class Timer():
	"""
	object to keep track of time
	"""
	def __init__(self):
		self.start_time = time.time()
		self.past_time = time.time()

	def __call__(self, *args, **kwargs):
		return self.print(*args, **kwargs)

	def print(self, string="", return_string=False):
		new_string = "%s Past Time: %f, Total Time: %f"%(string, 
			time.time() - self.past_time, time.time() - self.start_time)
		self.past_time = time.time()
		if return_string:
			return new_string
		print(new_string)

class StandardizedPaths():
	"""
	Class used for creating paths.
	Can add descriptions to paths on what they are supposed to to.
	
	These parameters will be saved into a paths config file which can used 
	to restore the parameters, or can be used to configure a new path.

	Paths won't be created unless the path is called/used or if the overwrite
	parameter is True.
	"""
	def __init__(self, base_path, config_filepath=None, overwrite=False, log_method=print):
		"""Initializes the paths. This will create base path if not found, and use/create the config file in base_path.
		Can use preexisting config file, but base path must not have one.
		
		Args:
		    base_path (str): the folder for this StandardizedPaths instance to use
		    config_filepath (str or None, optional): Separate configuration. None as default
		    overwrite (bool, optional): If there is an existing basepath, will delete it if this is True
		    log_method (function, optional): The way to log StandardizedPaths, could be file write, default is print
		"""
		# create base_path file if not exists.
		if not os.path.exists(base_path):
			log_method("Path not found...creating path.")
			os.makedirs(base_path)
		elif overwrite:
			shutil.rmtree(base_path)
			os.makedirs(base_path)

		base_path = os.path.realpath(base_path)

		config = cp.ConfigParser() #store data in config parser

		#if config file is not specified, create config file.
		if config_filepath is None:
			#try to find config file in path, ends with .path_cfg
			config_paths = glob.glob(os.path.join(base_path, "*.path_cfg"))
			assert len(config_paths) < 2, "Error, multiple config paths specified in directory."

			if not len(config_paths):
				config_filepath = os.path.join(base_path, "standardized_paths.path_cfg")
				log_method("Config file not found... creating file in path")
			else:
				config_filepath = config_paths[0]
				log_method("Config file found, using %s"%config_filepath)
		elif not os.path.exists(config_filepath):
			print("config file does not exist, creating new one in base_path")
			os.path.join(base_path, os.path.basename(config_filepath))
		else:
			config_filepath = os.path.realpath(config_filepath)
		

		if os.path.exists(config_filepath):
			config.read(config_filepath)
		
		# configuration details
		self.config = config # this will contain information about the path.
		self.base_path = base_path
		self.config_filepath = config_filepath

		# misc items to keep track of.
		self.log_method = log_method
		self.path_id = "Path"
		self.description_id = "Description"

	def __call__(self, name, path=None, description="None", is_overwrite=True):
		"""add and create path. return path.
		
		Args:
		    name (str): name of the path
		    path (str): The path
		    description (str, optional): The description of the path
		    is_overwrite (bool, optional): whether to overwrite the addition or not, will not delete the path.
		
		Returns:
		    str: The path
		"""
		if path is None:
			path = name
		self.add_path(name=name, path=path, description=description, is_overwrite=is_overwrite)
		return self.get_path(name=name)

	def get_path(self, name=None, return_path=True, return_desc=False, overwrite=False):
		"""
		gets path given name, to get list of available paths, leave no arguments
		to this function.
		

		"""
		#return available paths.
		if not (return_path or return_desc):
			print("must return something... returning path by default")
			return_path = True

		# return available options.
		available_names = self.config.sections()
		if name is None:
			return available_names

		assert name in available_names, "path name specified doesn't exist"
		
		path = self.config[name][self.path_id]
		if path == "./":
			path = ""
		path = os.path.join(self.base_path, path)
		
		description = self.config[name][self.description_id]

		if not os.path.exists(path):
			os.makedirs(path)
		elif overwrite:
			shutil.rmtree(path)
			os.makedirs(path)

		#return what was specified
		returned_items = []
		if return_path:
			returned_items.append(path)
		if return_desc:
			returned_items.append(description)

		if len(returned_items) == 1:
			return returned_items[0]
		else:
			return returned_items

	def add_path(self, name, path, description="None", is_overwrite=True):
		"""
		Add new path to configuration. Created only when get_path
		is called. Prevents unnecessary paths from being created.
		
		Args:
		    name (str): name of the path
		    path (str): relative/absolute filepath
		    description (str, optional): description of filepath
		"""
		if (not is_overwrite) and (name in self.config):
			return
		if path == "":
			path = "./"
		self.config[name] = {
			self.path_id:path,
			self.description_id:description}
		with open(self.config_filepath, "w") as f:
			self.config.write(f)
