#H5PY removed from the Array Class
#"Energy coefficient" inserted after Jonathan's analysis
#From 06/04/2017 on

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import h5py

out_path1 = 'D:\Documents\KMC\ASI'
out_path2 = 'E:\Matteo\Python\KMC\Data\output'

myG = '#00f700'
myB = '#0000ff'
myR = '#ff0000'
myY = '#f7f700'
myM = '#f700f7'
myK = '#000000'
myC = '#00f7f7'
myO = '#ffa500'
myGG = '#b0b0b0'

#-------------------- CLASS Definitions --------------------

#Container for an island composing the array
class BasicUnit:

	def __init__(self):
		self.address = np.zeros(2, dtype=np.int_)
		self.dir = 1 #1 is RIGHT/UP - 0 is LEFT/DOWN
		self.sign = 1 #Factor (0 or 1) for the choice of the probability: abs(x - factor)
		self.neigh_num = 6
		self.neigh_list = np.zeros(self.neigh_num, dtype=np.int_) - 1 #-1 means not used (it's not a valid linear address)
		self.neigh_sign = np.zeros(self.neigh_num, dtype=np.int_) #Factor (0 or 1) for modifying the direction: abs(x - factor)
		self.disorder = 0 #exponent correction for introducing disorder

#Container + Methods for working with the whole array
class Array:

	def __init__(self, input_list, input_prob, opt=None):
		if type(input_list) == list:
			self.standard_constructor(input_list, input_prob)
		else:
			self.object_constructor(input_list, input_prob, opt)
	
	def standard_constructor(self, input_list, input_prob):
		self.input_double = input_list[0] #File with DOUBLE Probs.
		self.input_single = input_list[1] #File with SINGLE Probs.
		self.input_attemptFreq = (input_list[2], input_list[3]) #Attempt frequency - (Double, Single) (Hz)
		self.input_energyCoeff = (input_list[4], input_list[5]) #Energy rescaling coeff. - (Double, Single)
		self.input_rows = input_list[6] #Vertices in each row
		self.input_cols = input_list[7] #Vertices in each column
		self.input_init = input_list[8] #Init type ('1', '2', '3', '4' or 'r')
		self.input_boundary = input_list[9] #Boundary conditions: 'finite' (default) or 'pbc'
		self.input_disorderStDev = input_list[10] #Standard Deviation for disorder
		self.input_kmcSteps = input_list[11] #Steps for each run
		self.input_timeLimit = input_list[12] #Time limit (s) for each run
		self.input_images = input_list[13] #Number of images to be saved
		self.input_run = -1 #Progressive run number
		
		#Generate related quantities
		self.freq_double = self.input_attemptFreq[0]*(input_prob[0]**self.input_energyCoeff[0])
		self.freq_single = self.input_attemptFreq[1]*(input_prob[1]**self.input_energyCoeff[1])
		if self.input_boundary == 'pbc': #pbc
			self.totEl = 2*self.input_rows*self.input_cols
		else: #finite
			self.totEl = self.input_rows*(self.input_cols+1) + self.input_cols*(self.input_rows+1)
		self.list = [BasicUnit() for i in range(self.totEl)] #The array as a collection of BasicUnit objects
		self.list_freq = np.zeros(self.totEl, dtype=np.float_)
		self.list_partialSum = np.zeros(self.totEl, dtype=np.float_)
		self.evolution = np.zeros((self.input_kmcSteps+1, 16), dtype=np.float_)
		self.evolution_T1 = np.zeros((self.input_kmcSteps+1, 2), dtype=np.float_)
		self.t = np.zeros(self.input_kmcSteps+1, dtype=np.float_)
		self.map = np.zeros((self.input_rows, self.input_cols), dtype=np.int_)
		self.index_saveImg = np.int_(np.round(np.power(10, np.linspace(0, np.log10(self.input_kmcSteps), self.input_images)))-1)
		self.index_saveImg = np.unique(self.index_saveImg)
		self.input_images = len(self.index_saveImg) + 1
		self.img_step = np.zeros(self.input_images, dtype=np.int_)
		self.img_archive = np.zeros((self.input_rows, self.input_cols, self.input_images), dtype=np.int_)
		
		#Methods for initializing the array
		self.generate_array()
		self.fill_array()
		self.identify_array()
	
	def object_constructor(self, old_obj, input_list, input_prob):
		self.input_double = input_list[0] #File with DOUBLE Probs.
		self.input_single = input_list[1] #File with SINGLE Probs.
		self.input_attemptFreq = old_obj.input_attemptFreq #Attempt frequency - [Double, Single] (Hz)
		self.input_energyCoeff = old_obj.input_energyCoeff #Energy rescaling coeff. - [Double, Single]
		self.input_rows = old_obj.input_rows #Vertices in each row
		self.input_cols = old_obj.input_cols #Vertices in each column
		self.input_init = old_obj.input_init #Init type ('1', '2', '3', '4' or 'r')
		self.input_boundary = old_obj.input_boundary #Boundary conditions: 'finite' (default) or 'pbc'
		self.input_disorderStDev = input_list[10] #Standard Deviation for disorder
		self.input_kmcSteps = input_list[11] #Steps for each run
		self.input_timeLimit = input_list[12] #Time limit (s) for each run
		self.input_images = input_list[13] #Number of images to be saved
		self.input_run = -1 #Progressive run number
		
		#Generate related quantities
		self.freq_double = self.input_attemptFreq[0]*(input_prob[0]**self.input_energyCoeff[0])
		self.freq_single = self.input_attemptFreq[1]*(input_prob[1]**self.input_energyCoeff[1])
		self.totEl = old_obj.totEl
		self.list = old_obj.list #The array as a collection of BasicUnit objects
		self.list_freq = np.zeros(self.totEl, dtype=np.float_)
		self.list_partialSum = np.zeros(self.totEl, dtype=np.float_)
		self.evolution = np.zeros((self.input_kmcSteps+1, 16), dtype=np.float_)
		self.evolution_T1 = np.zeros((self.input_kmcSteps+1, 2), dtype=np.float_)
		self.t = np.zeros(self.input_kmcSteps+1, dtype=np.float_)
		self.map = old_obj.map
		self.index_saveImg = np.int_(np.round(np.power(10, np.linspace(0, np.log10(self.input_kmcSteps), self.input_images)))-1)
		self.index_saveImg = np.unique(self.index_saveImg)
		self.input_images = len(self.index_saveImg) + 1
		self.img_step = np.zeros(self.input_images, dtype=np.int_)
		self.img_archive = np.zeros((self.input_rows, self.input_cols, self.input_images), dtype=np.int_)
		
		#Initialize the "new" evolution
		self.evolution[0, :] = old_obj.evolution[-1, :]*self.input_rows*self.input_cols
		self.evolution_T1[0, :] = old_obj.evolution_T1[-1, :]*self.input_rows*self.input_cols
		self.t[0] = old_obj.t[-1]
		
		#Initialize list_freq and list_partialSum
		for i in range(self.totEl):
			self.list_freq[i] = self.get_freq(i)
		self.list_partialSum[0] = self.list_freq[0]
		for i in range(self.totEl-1):
			self.list_partialSum[i+1] = self.list_partialSum[i] + self.list_freq[i+1]
		
	def get_address_linear(self, address_array):
		if self.input_boundary == 'pbc': #pbc
			output = self.input_cols*address_array[0] + address_array[1]
		else: #finite
			output = (self.input_cols+1)*np.floor(address_array[0]/2) + (self.input_cols)*np.ceil(address_array[0]/2) + address_array[1]
		return int(output)
		
	def wrap(self, address_array):
		if self.input_boundary == 'pbc':
			if address_array[0] == 2*self.input_rows:
				address_array[0] = 0
			elif address_array[0] < 0:
				address_array[0] = 2*self.input_rows + address_array[0]
			if address_array[1] == self.input_cols:
				address_array[1] = 0
			elif address_array[1] < 0:
				address_array[1] = self.input_cols + address_array[1]
		return address_array
	
	def get_vertexMask(self, starting_address):
		binary_map = np.zeros(4, dtype=np.int_)
		binary_map[0] = self.list[self.get_address_linear(self.wrap(starting_address))].dir
		following_address = [starting_address[0]+1, starting_address[1]]
		binary_map[1] = self.list[self.get_address_linear(self.wrap(following_address))].dir
		binary_map[1] = abs(binary_map[1] - 1)
		following_address = [starting_address[0], starting_address[1]+1]
		binary_map[2] = self.list[self.get_address_linear(self.wrap(following_address))].dir
		binary_map[2] = abs(binary_map[2] - 1)
		following_address = [starting_address[0]-1, starting_address[1]]
		binary_map[3] = self.list[self.get_address_linear(self.wrap(following_address))].dir
		return binary_map
	
	def extract_code(self, mask):
		power_array = np.power(2, np.flipud(np.arange(0, len(mask), dtype=np.int_)))
		return np.sum(mask*power_array)
	
	def get_freq(self, index):
		neigh_num = self.list[index].neigh_num
		binary_map = np.zeros(neigh_num, dtype=np.int_)
		for j in range(neigh_num):
			binary_map[j] = self.list[self.list[index].neigh_list[j]].dir
			binary_map[j] = abs(binary_map[j] - self.list[index].neigh_sign[j])
		row_index = self.extract_code(binary_map)
		col_index = abs(self.list[index].dir - self.list[index].sign)
		if neigh_num == 6:
			freq_out = self.freq_double[row_index, col_index]
		elif neigh_num == 3:
			freq_out = self.freq_single[row_index, col_index]
		else:
			freq_out = 0
			print('---> ERROR in initializing list_freq') #DEBUG control
		return np.power(freq_out, 1 + self.list[index].disorder) #Introducing disorder
	
	def get_event_index(self, prob):
		boundary = prob*self.list_partialSum[-1] #It should never be self.list_partialSum[-1]
		return np.sum(self.list_partialSum <= boundary)
		
	def update_evoT1(self, index, address, vertex_type, value):
		if vertex_type == 5:
			if ((address[0]+address[1])%2) == 0:
				self.evolution_T1[index, 0] += value
			else:
				self.evolution_T1[index, 1] += value
		elif vertex_type == 10:
			if ((address[0]+address[1])%2) == 0:
				self.evolution_T1[index, 1] += value
			else:
				self.evolution_T1[index, 0] += value
	
	#Update evolution and map
	def update_evo(self, index, itf):
		self.evolution[index+1, :] = self.evolution[index, :]
		self.evolution_T1[index+1, :] = self.evolution_T1[index, :]
		if self.list[itf].neigh_num == 6: #CENTRAL ARRAY or PBC
			if (self.list[itf].address[0]%2) == 0: #VERTICAL islands
				address1 = self.list[self.list[itf].neigh_list[2]].address #TOP
				pos_mask1 = 3
				address2 = self.list[self.list[itf].neigh_list[5]].address #BOTTOM
				pos_mask2 = 1
			else: #HORIZONTAL islands
				address1 = self.list[itf].address #RIGHT
				pos_mask1 = 0
				address2 = self.list[self.list[itf].neigh_list[1]].address #LEFT
				pos_mask2 = 2
			map_position1 = [np.int_((address1[0]-1)/2), address1[1]]
			map_position2 = [np.int_((address2[0]-1)/2), address2[1]]
			vertex_mask1 = self.get_vertexMask(address1)
			vertex_type1 = self.extract_code(vertex_mask1)
			vertex_mask2 = self.get_vertexMask(address2)
			vertex_type2 = self.extract_code(vertex_mask2)
			self.evolution[index+1, vertex_type1] -= 1
			self.evolution[index+1, vertex_type2] -= 1
			self.update_evoT1(index+1, map_position1, vertex_type1, -1)
			self.update_evoT1(index+1, map_position2, vertex_type2, -1)
			vertex_mask1[pos_mask1] = abs(vertex_mask1[pos_mask1] - 1)
			vertex_type1 = self.extract_code(vertex_mask1)
			vertex_mask2[pos_mask2] = abs(vertex_mask2[pos_mask2] - 1)
			vertex_type2 = self.extract_code(vertex_mask2)
			self.evolution[index+1, vertex_type1] += 1
			self.evolution[index+1, vertex_type2] += 1
			self.update_evoT1(index+1, map_position1, vertex_type1, 1)
			self.update_evoT1(index+1, map_position2, vertex_type2, 1)
			self.map[map_position1[0], map_position1[1]] = vertex_type1
			self.map[map_position2[0], map_position2[1]] = vertex_type2
		else: #Boundary islands for 'fixed' boundary conditions
			if self.list[itf].address[0] == 0:
				address1 = self.list[self.list[itf].neigh_list[0]].address #BOTTOM Row
				pos_mask1 = 3
			elif self.list[itf].address[0] == 2*self.input_rows:
				address1 = self.list[self.list[itf].neigh_list[2]].address #TOP Row
				pos_mask1 = 1
			elif self.list[itf].address[1] == 0:
				address1 = self.list[itf].address #LEFT Column
				pos_mask1 = 0
			else:
				address1 = self.list[self.list[itf].neigh_list[1]].address #RIGHT Column
				pos_mask1 = 2
			map_position1 = [np.int_((address1[0]-1)/2), address1[1]]
			vertex_mask1 = self.get_vertexMask(address1)
			vertex_type1 = self.extract_code(vertex_mask1)
			self.evolution[index+1, vertex_type1] -= 1
			self.update_evoT1(index+1, map_position1, vertex_type1, -1)
			vertex_mask1[pos_mask1] = abs(vertex_mask1[pos_mask1] - 1)
			vertex_type1 = self.extract_code(vertex_mask1)
			self.evolution[index+1, vertex_type1] += 1
			self.update_evoT1(index+1, map_position1, vertex_type1, 1)
			self.map[map_position1[0], map_position1[1]] = vertex_type1
	
	#Filling the attributes for each element
	def generate_array(self):
		
		#Filling the address list
		counter = 0
		if self.input_boundary == 'pbc': #pbc
			for i in range(2*self.input_rows):
				for j in range(self.input_cols):
					self.list[counter].address[0] = i
					self.list[counter].address[1] = j
					counter += 1
		else: #finite
			for i in range(2*self.input_rows+1):
				if (i%2) == 0: #VERTICAL islands
					for j in range(self.input_cols):
						self.list[counter].address[0] = i
						self.list[counter].address[1] = j
						counter += 1
				else: #HORIZONTAL islands
					for j in range(self.input_cols+1):
						self.list[counter].address[0] = i
						self.list[counter].address[1] = j
						counter += 1
		
		#Filling the neigh. list
		pbc_condition = self.input_boundary == 'pbc'
		for i in range(self.totEl):
			ext_rows = (self.list[i].address[0] == 0) or (self.list[i].address[0] == 2*self.input_rows)
			ext_cols = ((self.list[i].address[1] == 0) and (self.list[i].address[0]%2 != 0)) or (self.list[i].address[1] == self.input_cols)
			if pbc_condition or not (ext_rows or ext_cols): #CENTRAL ARRAY or PBC
				if (self.list[i].address[0]%2) == 0: #VERTICAL islands
					self.list[i].neigh_list[0] = self.get_address_linear(self.wrap([self.list[i].address[0]+2,self.list[i].address[1]]))
					self.list[i].neigh_list[1] = self.get_address_linear(self.wrap([self.list[i].address[0]-2,self.list[i].address[1]]))
					self.list[i].neigh_list[2] = self.get_address_linear(self.wrap([self.list[i].address[0]+1,self.list[i].address[1]]))
					self.list[i].neigh_sign[2] = 1
					self.list[i].neigh_list[3] = self.get_address_linear(self.wrap([self.list[i].address[0]+1,self.list[i].address[1]+1]))
					self.list[i].neigh_sign[3] = 1
					self.list[i].neigh_list[4] = self.get_address_linear(self.wrap([self.list[i].address[0]-1,self.list[i].address[1]+1]))
					self.list[i].neigh_sign[4] = 1
					self.list[i].neigh_list[5] = self.get_address_linear(self.wrap([self.list[i].address[0]-1,self.list[i].address[1]]))
					self.list[i].neigh_sign[5] = 1
				else: #HORIZONTAL islands
					self.list[i].neigh_list[0] = self.get_address_linear(self.wrap([self.list[i].address[0],self.list[i].address[1]+1]))
					self.list[i].neigh_list[1] = self.get_address_linear(self.wrap([self.list[i].address[0],self.list[i].address[1]-1]))
					self.list[i].neigh_list[2] = self.get_address_linear(self.wrap([self.list[i].address[0]+1,self.list[i].address[1]]))
					self.list[i].neigh_list[3] = self.get_address_linear(self.wrap([self.list[i].address[0]-1,self.list[i].address[1]]))
					self.list[i].neigh_list[4] = self.get_address_linear(self.wrap([self.list[i].address[0]-1,self.list[i].address[1]-1]))
					self.list[i].neigh_list[5] = self.get_address_linear(self.wrap([self.list[i].address[0]+1,self.list[i].address[1]-1]))
			else: #Boundary islands for 'fixed' boundary conditions
				self.list[i].neigh_num = 3
				if ext_rows:
					if self.list[i].address[0] == 0:
						self.list[i].neigh_list[0] = self.get_address_linear([self.list[i].address[0]+1,self.list[i].address[1]])
						self.list[i].neigh_list[1] = self.get_address_linear([self.list[i].address[0]+2,self.list[i].address[1]])
						self.list[i].neigh_sign[1] = 1
						self.list[i].neigh_list[2] = self.get_address_linear([self.list[i].address[0]+1,self.list[i].address[1]+1])
						self.list[i].neigh_sign[2] = 1
					else:
						self.list[i].sign = 0
						self.list[i].neigh_list[0] = self.get_address_linear([self.list[i].address[0]-1,self.list[i].address[1]+1])
						self.list[i].neigh_sign[0] = 1
						self.list[i].neigh_list[1] = self.get_address_linear([self.list[i].address[0]-2,self.list[i].address[1]])
						self.list[i].neigh_list[2] = self.get_address_linear([self.list[i].address[0]-1,self.list[i].address[1]])
				else:
					if self.list[i].address[1] == 0:
						self.list[i].neigh_list[0] = self.get_address_linear([self.list[i].address[0]+1,self.list[i].address[1]])
						self.list[i].neigh_sign[0] = 1
						self.list[i].neigh_list[1] = self.get_address_linear([self.list[i].address[0],self.list[i].address[1]+1])
						self.list[i].neigh_sign[1] = 1
						self.list[i].neigh_list[2] = self.get_address_linear([self.list[i].address[0]-1,self.list[i].address[1]])
					else:
						self.list[i].sign = 0
						self.list[i].neigh_list[0] = self.get_address_linear([self.list[i].address[0]-1,self.list[i].address[1]-1])
						self.list[i].neigh_list[1] = self.get_address_linear([self.list[i].address[0],self.list[i].address[1]-1])
						self.list[i].neigh_list[2] = self.get_address_linear([self.list[i].address[0]+1,self.list[i].address[1]-1])
						self.list[i].neigh_sign[2] = 1
		
		#Filling the disorder (Gaussian)
		rand_disorder = self.input_disorderStDev*np.random.randn(self.totEl)
		for i in range(self.totEl):
			self.list[i].disorder = rand_disorder[i]
				
	#Filling the array with the initial state ('2' is the default status)
	def fill_array(self):
		if (not (self.input_boundary == 'pbc')) or ((self.input_rows%2) == 0 and (self.input_cols%2) == 0):
			if self.input_init == '1':
				for i in range(self.totEl):
					if (self.list[i].address[0]%4) == 0:
						if (self.list[i].address[1]%2) != 0:
							self.list[i].dir = 0
					elif (self.list[i].address[0]%2) == 0:
						if (self.list[i].address[1]%2) == 0:
							self.list[i].dir = 0
					else:
						if ((self.list[i].address[0]-1)%4) == 0:
							if (self.list[i].address[1]%2) == 0:
								self.list[i].dir = 0
						elif ((self.list[i].address[0]-1)%2) == 0:
							if (self.list[i].address[1]%2) != 0:
								self.list[i].dir = 0
			elif self.input_init == '3':
				for i in range(self.totEl):
					if (self.list[i].address[0]%2) != 0:
						if (self.list[i].address[1]%2) != 0:
							self.list[i].dir = 0
			if self.input_init == '4':
				for i in range(self.totEl):
					if (self.list[i].address[0]%4) == 0:
						if (self.list[i].address[1]%2) != 0:
							self.list[i].dir = 0
					elif (self.list[i].address[0]%2) == 0:
						if (self.list[i].address[1]%2) == 0:
							self.list[i].dir = 0
					else:
						if ((self.list[i].address[0]-1)%4) == 0:
							if (self.list[i].address[1]%2) != 0:
								self.list[i].dir = 0
						elif ((self.list[i].address[0]-1)%2) == 0:
							if (self.list[i].address[1]%2) == 0:
								self.list[i].dir = 0
		if self.input_init == 'r':
			rand_gen = np.random.randint(2, size=self.totEl)
			for i in range(self.totEl):
				self.list[i].dir = rand_gen[i]

	#Count the vertices and initialize the map - initialize list_freq and list_partialSum
	def identify_array(self):
		#Count the vertices and initialize the map
		for i in range(self.input_rows):
			for j in range(self.input_cols):
				vertex_type = self.extract_code(self.get_vertexMask([2*i+1, j]))
				self.evolution[0, vertex_type] += 1
				self.update_evoT1(0, [i, j], vertex_type, 1)
				self.map[i, j] = vertex_type
		
		#Initialize list_freq and list_partialSum
		for i in range(self.totEl):
			self.list_freq[i] = self.get_freq(i)
		self.list_partialSum[0] = self.list_freq[0]
		for i in range(self.totEl-1):
			self.list_partialSum[i+1] = self.list_partialSum[i] + self.list_freq[i+1]
	
	#Time evolution (for kmcSteps) and saving the output 
	def run(self, run_id=-1):
		if run_id != -1:
			self.input_run = run_id
		rand_island = np.random.random_sample(self.input_kmcSteps)
		rand_time = np.random.random_sample(self.input_kmcSteps)
		img_counter = 0
		self.img_archive[:, :, img_counter] = np.copy(self.map) #record starting image
		oldProgress = 0
		for i in np.arange(self.input_kmcSteps):
			#Progress
			progress = np.floor(100.0*i/self.input_kmcSteps)
			if progress != oldProgress:
				print('Run {:d} - Progress: {:.1f}%'.format(self.input_run, progress))
				oldProgress = progress
			#Identify the island to be flipped
			island_to_flip = self.get_event_index(rand_island[i])
			#Update evo and map
			self.update_evo(i, island_to_flip)
			#Flip the island in list
			self.list[island_to_flip].dir = abs(self.list[island_to_flip].dir - 1)
			#Update list_freq
			self.list_freq[island_to_flip] = self.get_freq(island_to_flip)
			for j in range(self.list[island_to_flip].neigh_num):
				self.list_freq[self.list[island_to_flip].neigh_list[j]] = self.get_freq(self.list[island_to_flip].neigh_list[j])
			#Update list_partialSum
			self.list_partialSum[0] = self.list_freq[0]
			for j in range(self.totEl-1):
				self.list_partialSum[j+1] = self.list_partialSum[j] + self.list_freq[j+1]
			#Update t
			self.t[i+1] = self.t[i] - np.log(rand_time[i])/self.list_partialSum[-1]
			#Save image (with logarithmic sampling)
			if np.sum(self.index_saveImg == i) >= 1:
				img_counter += 1
				self.img_archive[:, :, img_counter] = np.copy(self.map) #record following images
				self.img_step[img_counter] = i + 1
		self.input_timeLimit = self.t[-1] - self.t[0]
		self.evolution /= self.input_rows*self.input_cols
		self.evolution_T1 /= self.input_rows*self.input_cols

#-------------------- SAVE Function --------------------

#Save the time evolution and the images (with attributes)
#No more part of the Array Class
def save_evolution(f, obj):
	dset_evo_name = 'run{:d}/evo'.format(obj.input_run)
	dset_evo = f.create_dataset(dset_evo_name, data=obj.evolution)
	dset_evo.attrs['input_double'] = obj.input_double
	dset_evo.attrs['input_single'] = obj.input_single
	dset_evo.attrs['dim'] = (obj.input_rows, obj.input_cols)
	dset_evo.attrs['init'] = obj.input_init
	dset_evo.attrs['boundary'] = obj.input_boundary
	dset_evo.attrs['kmcSteps'] = obj.input_kmcSteps
	dset_evo.attrs['timeLimit'] = obj.input_timeLimit
	dset_evo.attrs['images'] = obj.input_images
	dset_evo.attrs['attempt_freq'] = obj.input_attemptFreq
	dset_evo.attrs['energy_coeff'] = obj.input_energyCoeff
	dset_evo.attrs['disorder'] = obj.input_disorderStDev
	
	dset_t_name = 'run{:d}/t'.format(obj.input_run)
	dset_t = f.create_dataset(dset_t_name, data=obj.t)
	
	dset_evoT1_name = 'run{:d}/evo_T1'.format(obj.input_run)
	dset_evoT1 = f.create_dataset(dset_evoT1_name, data=obj.evolution_T1)
	
	dset_doubleFreq_name = 'run{:d}/f_double'.format(obj.input_run)
	dset_doubleFreq = f.create_dataset(dset_doubleFreq_name, data=obj.freq_double)
	dset_doubleFreq.attrs['input_double'] = obj.input_double
	dset_doubleFreq.attrs['attempt_freq'] = obj.input_attemptFreq[0]
	dset_doubleFreq.attrs['energy_coeff'] = obj.input_energyCoeff[0]
	
	dset_singleFreq_name = 'run{:d}/f_single'.format(obj.input_run)
	dset_singleFreq = f.create_dataset(dset_singleFreq_name, data=obj.freq_single)
	dset_singleFreq.attrs['input_single'] = obj.input_single
	dset_singleFreq.attrs['attempt_freq'] = obj.input_attemptFreq[1]
	dset_singleFreq.attrs['energy_coeff'] = obj.input_energyCoeff[1]
	
	for i in range(obj.input_images):
		dset_image_name = 'run{:d}/images/img{:d}'.format(obj.input_run, i)
		dset_img = f.create_dataset(dset_image_name, data=obj.img_archive[:, :, i])
		dset_img.attrs['t'] = obj.t[obj.img_step[i]]
		dset_img.attrs['step'] = obj.img_step[i]

#-------------------- ANALYSIS Functions --------------------

#Average all the runs in a given HDF5 file
def avg_runs(f):
	if 'avg' in f.keys():
		del f['avg']
		print('AVG group deleted')
	num_runs = len(f.keys())
	#Get the minimum (first) end time and the corresponding run
	t_min = f['run0/t'].value[-1]
	run_min = 0
	for run in range(num_runs):
		dset_t_name = 'run{:d}/t'.format(run)
		t_value = f[dset_t_name].value[-1]
		if t_value < t_min:
			t_min = t_value
			run_min = run
	dset_t_name = 'run{:d}/t'.format(run_min)
	t_avg = f[dset_t_name].value #The time trace corresponding to the (first) minimum end time
	
	#Interpolating and averaging
	kmcSteps = f['run0/evo'].attrs['kmcSteps']
	evolution_avg = np.zeros((kmcSteps+1, 16), dtype=np.float_)
	evolutionT1_avg = np.zeros((kmcSteps+1, 2), dtype=np.float_)
	for run in range(num_runs):
		evolution_run = f['run{:d}/evo'.format(run)].value
		evolutionT1_run = f['run{:d}/evo_T1'.format(run)].value
		t_run = f['run{:d}/t'.format(run)].value
		for i in range(16):
			evolution_avg[:, i] += np.interp(t_avg, t_run, evolution_run[:, i])
		for i in range(2):
			evolutionT1_avg[:, i] += np.interp(t_avg, t_run, evolutionT1_run[:, i])
	
	#Save the avg run in the same HDF5 file
	dset_evo_avg = f.create_dataset('avg/evo', data=evolution_avg/num_runs)
	dset_evo_avg.attrs['input_double'] = f['run0/evo'].attrs['input_double']
	dset_evo_avg.attrs['input_single'] = f['run0/evo'].attrs['input_single']
	dset_evo_avg.attrs['boundary'] = f['run0/evo'].attrs['boundary']
	dset_evo_avg.attrs['dim'] = f['run0/evo'].attrs['dim']
	dset_evo_avg.attrs['init'] = f['run0/evo'].attrs['init']
	dset_evo_avg.attrs['attempt_freq'] = f['run0/evo'].attrs['attempt_freq']
	dset_evo_avg.attrs['energy_coeff'] = f['run0/evo'].attrs['energy_coeff']
	dset_evo_avg.attrs['disorder'] = f['run0/evo'].attrs['disorder']
	dset_evo_avg.attrs['kmcSteps'] = kmcSteps
	dset_evo_avg.attrs['timeLimit'] = t_avg[-1]
	dset_evo_avg.attrs['runs_for_avg'] = num_runs
	dset_t_avg = f.create_dataset('avg/t', data=t_avg)
	dset_evoT1_avg = f.create_dataset('avg/evo_T1', data=evolutionT1_avg/num_runs)
	dset_doubleFreq_avg = f.create_dataset('avg/f_double', data=f['run0/f_double'].value)
	dset_doubleFreq_avg.attrs['input_double'] = f['run0/f_double'].attrs['input_double']
	dset_doubleFreq_avg.attrs['attempt_freq'] = f['run0/f_double'].attrs['attempt_freq']
	dset_doubleFreq_avg.attrs['energy_coeff'] = f['run0/f_double'].attrs['energy_coeff']
	dset_singleFreq_avg = f.create_dataset('avg/f_single', data=f['run0/f_single'].value)
	dset_singleFreq_avg.attrs['input_single'] = f['run0/f_single'].attrs['input_single']
	dset_singleFreq_avg.attrs['attempt_freq'] = f['run0/f_single'].attrs['attempt_freq']
	dset_singleFreq_avg.attrs['energy_coeff'] = f['run0/f_single'].attrs['energy_coeff']
	print('AVG group created ({:d} runs)'.format(num_runs))

#Merge ALL the runs in a given HDF5 file
def merge_runs(f):
	num_runs = len(f.keys())
	kmcSteps = np.zeros(num_runs, dtype=np.float_)
	for run in range(num_runs):
		kmcSteps[run] = f['run{:d}/evo'.format(run)].attrs['kmcSteps']
	kmcSteps_tot = np.sum(kmcSteps)
	
	evolution_merged = np.zeros((kmcSteps_tot+1, 16), dtype=np.float_)
	evolutionT1_merged = np.zeros((kmcSteps_tot+1, 2), dtype=np.float_)
	t_merged = np.zeros(kmcSteps_tot+1, dtype=np.float_)
	evolution_run = f['run0/evo'].value
	evolutionT1_run = f['run0/evo_T1'].value
	t_run = f['run0/t'].value
	evolution_merged[0:kmcSteps[0]+1, :] = evolution_run
	evolutionT1_merged[0:kmcSteps[0]+1, :] = evolutionT1_run
	t_merged[0:kmcSteps[0]+1] = t_run
	start_index = kmcSteps[0] + 1
	for run in range(num_runs-1):
		stop_index = start_index + kmcSteps[run+1]
		evolution_run = f['run{:d}/evo'.format(run+1)].value
		evolutionT1_run = f['run{:d}/evo_T1'.format(run+1)].value
		t_run = f['run{:d}/t'.format(run+1)].value
		evolution_merged[start_index:stop_index, :] = evolution_run[1:, :]
		evolutionT1_merged[start_index:stop_index, :] = evolutionT1_run[1:, :]
		t_merged[start_index:stop_index] = t_run[1:]
		start_index = stop_index
		
	#Save the merged run in the same HDF5 file
	dset_evo_merged = f.create_dataset('merged/evo', data=evolution_merged)
	dset_evo_merged.attrs['boundary'] = f['run0/evo'].attrs['boundary']
	dset_evo_merged.attrs['dim'] = f['run0/evo'].attrs['dim']
	dset_evo_merged.attrs['init'] = f['run0/evo'].attrs['init']
	dset_evo_merged.attrs['attempt_freq'] = f['run0/evo'].attrs['attempt_freq']
	dset_evo_merged.attrs['energy_coeff'] = f['run0/evo'].attrs['energy_coeff']
	dset_evo_merged.attrs['disorder'] = f['run0/evo'].attrs['disorder']
	dset_evo_merged.attrs['kmcSteps'] = kmcSteps_tot
	dset_evo_merged.attrs['timeLimit'] = t_merged[-1]
	dset_evo_merged.attrs['merged_runs'] = num_runs
	dset_t_merged = f.create_dataset('merged/t', data=t_merged)
	dset_evoT1_merged = f.create_dataset('merged/evo_T1', data=evolutionT1_merged)
	print('MERGED group created ({:d} runs)'.format(num_runs))

#-------------------- DRAWING Functions --------------------

#Draw colored map (given the run and the image num.)
def draw_map(input_name, run, image, type='v', file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	if image == -1:
		image = len(f['run{:d}/images'.format(run)]) - 1
	dset_name = 'run{:d}/images/img{:d}'.format(run, image)
	map = f[dset_name].value
	t = f[dset_name].attrs['t']
	step = f[dset_name].attrs['step']
	if file_flag:
		f.close()
	
	if type == 'v':
		color_dictionary = [myY,myR,myR,myB,myR,myG,myB,myR,myR,myB,myG,myR,myB,myR,myR,myY]
	elif type == '1':
		color_dictionary = [myK,myK,myK,myK,myK,myG,myK,myK,myK,myK,myM,myK,myK,myK,myK,myK]
	elif type == '2':
		color_dictionary = [myK,myK,myK,myB,myK,myK,myC,myK,myK,myO,myK,myK,myGG,myK,myK,myK]
	elif type == '3':
		color_dictionary = [myK,myB,myR,myK,myY,myK,myK,myM,myC,myK,myK,myO,myK,myGG,myG,myK]
	elif type == '4':
		color_dictionary = [myY,myK,myK,myK,myK,myK,myK,myK,myK,myK,myK,myK,myK,myK,myK,myO]
	elif type == 'p':
		color_dictionary = [myG,myM,myK]
	else:
		color_dictionary = [myY,myR,myR,myB,myR,myG,myB,myR,myR,myB,myG,myR,myB,myR,myR,myY]
	rows, cols = map.shape
	x_coord = [0]*rows*cols
	y_coord = [0]*rows*cols
	color = ['']*rows*cols
	counter = 0
	for i in range(rows):
		for j in range(cols):
			x_coord[counter] = j
			y_coord[counter] = i
			if type == 'p':
				if map[i,j] == 5:
					if ((i+j)%2) == 0:
						color[counter] = color_dictionary[0]
					else:
						color[counter] = color_dictionary[1]
				elif map[i,j] == 10:
					if ((i+j)%2) == 0:
						color[counter] = color_dictionary[1]
					else:
						color[counter] = color_dictionary[0]
				else:
					color[counter] = color_dictionary[2]
			else:
				color[counter] = color_dictionary[map[i,j]]
			counter += 1
	
	size_points = 20e4*(1/max(rows, cols))**2
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_coord, y_coord, size_points, color, edgecolors=color)
	plt.title('{:d} x {:d} Vertices - Run {:d}, Image {:d}, Step {:.1f} - t = {:.4e} s'.format(rows, cols, run, image, step, t))
	ax.axis('scaled')
	ax.set_xlim([-1, cols])
	ax.set_ylim([-1, rows])
	plt.show()

#Plot the 4 vertices evolution (given the run)
def plot_evo(input_name, run, image_flag=False, file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	if run == -1:
		dset_evo_name = 'avg/evo'
		dset_t_name = 'avg/t'
		multipleRuns = f[dset_evo_name].attrs['runs_for_avg']
	elif run == -2:
		dset_evo_name = 'merged/evo'
		dset_t_name = 'merged/t'
		multipleRuns = f[dset_evo_name].attrs['merged_runs']
	else:
		dset_evo_name = 'run{:d}/evo'.format(run)
		dset_t_name = 'run{:d}/t'.format(run)
	evo = f[dset_evo_name].value[1:, :] #initial value skipped because of LOG plot (t=0 is not drawable)
	t = f[dset_t_name].value[1:] #initial value skipped because of LOG plot (t=0 is not drawable)
	kmcSteps = f[dset_evo_name].attrs['kmcSteps']
	rows, cols = f[dset_evo_name].attrs['dim']
	if image_flag and run >= 0:
		images_num = f[dset_evo_name].attrs['images']
		x_coord = np.zeros(images_num-1, dtype=np.float_)
		for i in range(images_num-1):
			x_coord[i] = f['run{:d}/images/img{:d}'.format(run, i+1)].attrs['t']
	if file_flag:
		f.close()
	
	evo_dictionary_4vert = [3,2,2,1,2,0,1,2,2,1,0,2,1,2,2,3]
	color_dictionary_4vert = [myG, myB, myR, myY]
	evo_4vertices = np.zeros((kmcSteps, 4), dtype=np.float_)
	for i in range(16):
		evo_4vertices[:, evo_dictionary_4vert[i]] += evo[:, i]
	
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	for i in range(4):
		ax.semilogx(t, evo_4vertices[:,i], '-', color=color_dictionary_4vert[i], linewidth=2, label='T{:d}'.format(i+1))
		#ax.plot(t, evo_4vertices[:,i], '-', color=color_dictionary_4vert[i], linewidth=2, label='T{:d}'.format(i+1))
	if run == -1 or run == -2:
		plt.title('{:d} x {:d} Vertices - {:d} Runs considered - {:.1f} KMC Steps'.format(rows, cols, multipleRuns, kmcSteps))
	else:
		plt.title('{:d} x {:d} Vertices - Run {:d} - {:.1f} KMC Steps'.format(rows, cols, run, kmcSteps))
	plt.xlabel('t (s)')
	plt.ylabel('P(t)')
	ax.set_ylim([0, 1])
	if image_flag and run != -1:
		for i in range(images_num-1):
			ax.axvline(x_coord[i], color='k')
	ax.grid(True)
	ax.legend(loc='best')
	plt.show()
	return ax
	
#Plot the T1 phases evolution (given the run)
def plot_evoT1(input_name, run, image_flag=False, file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	if run == -1:
		dset_evo_name = 'avg/evo'
		dset_evoT1_name = 'avg/evo_T1'
		dset_t_name = 'avg/t'
		multipleRuns = f[dset_evo_name].attrs['runs_for_avg']
	elif run == -2:
		dset_evo_name = 'merged/evo'
		dset_evoT1_name = 'merged/evo_T1'
		dset_t_name = 'merged/t'
		multipleRuns = f[dset_evo_name].attrs['merged_runs']
	else:
		dset_evo_name = 'run{:d}/evo'.format(run)
		dset_evoT1_name = 'run{:d}/evo_T1'.format(run)
		dset_t_name = 'run{:d}/t'.format(run)
	evo = f[dset_evo_name].value[1:, :] #initial value skipped because of LOG plot (t=0 is not drawable)
	evo_T1 = f[dset_evoT1_name].value[1:, :] #initial value skipped because of LOG plot (t=0 is not drawable)
	t = f[dset_t_name].value[1:] #initial value skipped because of LOG plot (t=0 is not drawable)
	kmcSteps = f[dset_evo_name].attrs['kmcSteps']
	rows, cols = f[dset_evo_name].attrs['dim']
	if image_flag and run >= 0:
		images_num = f[dset_evo_name].attrs['images']
		x_coord = np.zeros(images_num-1, dtype=np.float_)
		for i in range(images_num-1):
			x_coord[i] = f['run{:d}/images/img{:d}'.format(run, i+1)].attrs['t']
	if file_flag:
		f.close()
	
	color_dictionary_T1 = [myG, myM, myK]
	evo_CompleteT1 = np.zeros((kmcSteps, 3), dtype=np.float_)
	evo_CompleteT1[:, 0:2] = evo_T1
	for i in range(16):
		if (i != 5) and (i != 10):
			evo_CompleteT1[:, 2] += evo[:, i]
	
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	for i in range(2):
		ax.semilogx(t, evo_CompleteT1[:,i], '-', color=color_dictionary_T1[i], linewidth=2, label='T1 Phase {:d}'.format(i))
	ax.semilogx(t, evo_CompleteT1[:,2], '-', color=color_dictionary_T1[2], linewidth=2, label='Boundary')
	if run == -1 or run == -2:
		plt.title('{:d} x {:d} Vertices - {:d} Runs considered - {:.1f} KMC Steps'.format(rows, cols, multipleRuns, kmcSteps))
	else:
		plt.title('{:d} x {:d} Vertices - Run {:d} - {:.1f} KMC Steps'.format(rows, cols, run, kmcSteps))
	plt.xlabel('t (s)')
	plt.ylabel('P(t)')
	ax.set_ylim([0, 1])
	if image_flag and run != -1:
		for i in range(images_num-1):
			ax.axvline(x_coord[i], color='k')
	ax.grid(True)
	ax.legend(loc='best')
	plt.show()

#Plot the 4 vertices MASTER EQUATION evolution given the run and optionally the axis (Simple DVA M. Eq.)
def plot_meq(input_name, run, ax='', noT1=(False, ''), file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	if run == -1:
		dset_evo_name = 'avg/evo'
		dset_doubleFreq_name = 'avg/f_double'
		dset_t_name = 'avg/t'
	else:
		dset_evo_name = 'run{:d}/evo'.format(run)
		dset_doubleFreq_name = 'run{:d}/f_double'.format(run)
		dset_t_name = 'run{:d}/t'.format(run)
	evo = f[dset_evo_name].value
	doubleFreq = f[dset_doubleFreq_name].value
	t = f[dset_t_name].value
	kmcSteps = f[dset_evo_name].attrs['kmcSteps']
	if file_flag:
		f.close()
	
	evo_dictionary_4vert = [3,2,2,1,2,0,1,2,2,1,0,2,1,2,2,3]
	evo_4vertices = np.zeros((kmcSteps+1, 4), dtype=np.float_)
	for i in range(16):
		evo_4vertices[:, evo_dictionary_4vert[i]] += evo[:, i]
	y0_4 = evo_4vertices[0, :] #starting state (4 vertices)
	
	#- DICTIONARY -
	# 0 - 11
	# 1 - 12
	# 2 - 13
	# 3 - 14
	# 4 - 22
	# 5 - 23
	# 6 - 24
	# 7 - 33
	# 8 - 34
	# 9 - 44
	#-------------
	freq_table = np.zeros((10, 10), dtype=np.float_)
	freq_table[7, 0] = 16*doubleFreq[10, 0]
	freq_table[7, 3] = 32*doubleFreq[6, 0]
	freq_table[7, 1] = 32*doubleFreq[17, 1]
	freq_table[5, 2] = 16*doubleFreq[2, 0]
	freq_table[8, 2] = 8*doubleFreq[21, 1]
	freq_table[7, 4] = 8*(doubleFreq[0, 1] + doubleFreq[3, 1])
	freq_table[7, 6] = 32*doubleFreq[20, 0]
	freq_table[8, 5] = 8*doubleFreq[4, 1]
	freq_table[7, 9] = 16*doubleFreq[5, 0]
	#Reverse transitions
	freq_table[0, 7] = doubleFreq[10, 1]
	freq_table[3, 7] = 2*doubleFreq[6, 1]
	freq_table[1, 7] = 4*doubleFreq[17, 0]
	freq_table[2, 5] = 8*doubleFreq[2, 1]
	freq_table[2, 8] = 8*doubleFreq[21, 0]
	freq_table[4, 7] = 2*(doubleFreq[0, 0] + doubleFreq[3, 0])
	freq_table[6, 7] = 4*doubleFreq[20, 1]
	freq_table[5, 8] = 16*doubleFreq[4, 0]
	freq_table[9, 7] = doubleFreq[5, 1]
	freq_table /= 8 #global factor
	#Derivative function
	def meq_func(y, t, f):
		dyAnn = np.zeros(4, dtype=np.float_)
		dyCre = np.zeros(4, dtype=np.float_)
		dyAnn[0] = 2*y[0]*y[0]*f[7,0] + y[0]*y[3]*f[7,3] + y[0]*y[1]*f[7,1] + y[0]*y[2]*(f[5,2] + f[8,2])
		dyCre[0] = y[1]*y[2]*f[2,5] + y[3]*y[2]*f[2,8] + y[2]*y[2]*(2*f[0,7] + f[3,7] + f[1,7])
		dyAnn[1] = 2*y[1]*y[1]*f[7,4] + y[1]*y[0]*f[7,1] + y[1]*y[3]*f[7,6] + y[1]*y[2]*(f[2,5] + f[8,5])
		dyCre[1] = y[0]*y[2]*f[5,2] + y[3]*y[2]*f[5,8] + y[2]*y[2]*(2*f[4,7] + f[1,7] + f[6,7])
		dyAnn[2] = 2*y[2]*y[2]*(f[0,7] + f[4,7] + f[9,7] + f[3,7] + f[1,7] + f[6,7])
		dyCre[2] = 2*(y[0]*y[0]*f[7,0] + y[0]*y[3]*f[7,3] + y[0]*y[1]*f[7,1] + y[1]*y[1]*f[7,4] + y[1]*y[3]*f[7,6] + y[3]*y[3]*f[7,9])
		dyAnn[3] = 2*y[3]*y[3]*f[7,9] + y[3]*y[1]*f[7,6] + y[3]*y[0]*f[7,3] + y[3]*y[2]*(f[2,8] + f[5,8])
		dyCre[3] = y[0]*y[2]*f[8,2] + y[1]*y[2]*f[8,5] + y[2]*y[2]*(2*f[9,7] + f[3,7] + f[6,7])
		return (dyCre - dyAnn)
	#Derivative function - No T1 ( y[0] -> c*(y[1] + y[2] + y[3]) )
	#Also y[3] could be added, but when domain forms, y[3] has already disappeared
	def meq_func_noT1(y, t, f, c):
		dyAnn = np.zeros(4, dtype=np.float_)
		dyCre = np.zeros(4, dtype=np.float_)
		#no y[0] evolution
		my_y0 = c*(y[1] + y[2] + y[3])
		dyAnn[1] = 2*y[1]*y[1]*f[7,4] + y[1]*my_y0*f[7,1] + y[1]*y[3]*f[7,6] + y[1]*y[2]*(f[2,5] + f[8,5])
		dyCre[1] = my_y0*y[2]*f[5,2] + y[3]*y[2]*f[5,8] + y[2]*y[2]*(2*f[4,7] + f[1,7] + f[6,7])
		dyAnn[2] = 2*y[2]*y[2]*(f[0,7] + f[4,7] + f[9,7] + f[3,7] + f[1,7] + f[6,7])
		dyCre[2] = 2*(my_y0*my_y0*f[7,0] + my_y0*y[3]*f[7,3] + my_y0*y[1]*f[7,1] + y[1]*y[1]*f[7,4] + y[1]*y[3]*f[7,6] + y[3]*y[3]*f[7,9])
		dyAnn[3] = 2*y[3]*y[3]*f[7,9] + y[3]*y[1]*f[7,6] + y[3]*my_y0*f[7,3] + y[3]*y[2]*(f[2,8] + f[5,8])
		dyCre[3] = my_y0*y[2]*f[8,2] + y[1]*y[2]*f[8,5] + y[2]*y[2]*(2*f[9,7] + f[3,7] + f[6,7])
		return (dyCre - dyAnn)
	#Solve the Diff. Eq. System
	if noT1[0]:
		y_sol = odeint(meq_func_noT1, y0_4, t, args=(freq_table, noT1[1]))
		y_sol[:,0] = 1 - (y_sol[:,1] + y_sol[:,2] + y_sol[:,3])
	else:
		y_sol = odeint(meq_func, y0_4, t, args=(freq_table,))
	
	color_dictionary_4vert = [myM, myC, myK, myO]
	if ax == '':
		fig = plt.figure(figsize=(12,12))
		ax = fig.add_subplot(1,1,1)
	for i in range(4):
		if noT1[0]:
			label_text = 'T{:d} - M. Eq. {:.2f}'.format(i+1, noT1[1])
		else:
			label_text = 'T{:d} - M. Eq.'.format(i+1)
		#initial value skipped because of LOG plot (t=0 is not drawable)
		ax.semilogx(t[1:], y_sol[1:, i], '--', color=color_dictionary_4vert[i], linewidth=2, label=label_text)
	plt.xlabel('t (s)')
	plt.ylabel('P(t)')
	ax.set_ylim([0, 1])
	ax.grid(True)
	ax.legend(loc='best')
	
	if not(noT1[0]):
		evo_diff_meq = evo_4vertices - y_sol
		fig_diff = plt.figure(figsize=(12,12))
		ax_diff = fig_diff.add_subplot(1,1,1)
		for i in range(4):
			#initial value skipped because of LOG plot (t=0 is not drawable)
			ax_diff.semilogx(t[1:], evo_diff_meq[1:, i], '-', color=color_dictionary_4vert[i], linewidth=2, label='T{:d} - Diff.'.format(i+1))
		ax_diff.axhline(0, color='k')
		plt.title('Evolution vs M. Eq. for ' + ax.get_title())
		plt.xlabel('t (s)')
		plt.ylabel('P$_{Evo}$(t) - P$_{MEq}$(t)')
		#ax_diff.set_ylim([0, 1])
		ax_diff.grid(True)
		ax_diff.legend(loc='best')
	
	plt.show()
	return y0_4

#Histogram of dt (and hopefully fit)
def time_stats(input_name, run, num_bins=100, file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	if run == -1:
		dset_evo_name = 'avg/evo'
		dset_t_name = 'avg/t'
	elif run == -2:
		dset_evo_name = 'merged/evo'
		dset_t_name = 'merged/t'
	else:
		dset_evo_name = 'run{:d}/evo'.format(run)
		dset_t_name = 'run{:d}/t'.format(run)
	timeLimit = f[dset_evo_name].attrs['timeLimit']
	t = f[dset_t_name].value #The whole time trace
	if file_flag:
		f.close()
	
	t_diff = t[1:] - t[:-1]
	num_data = len(t_diff)
	
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(1,1,1)
	n, bins, patches = ax.hist(t_diff, num_bins, facecolor='green', alpha=0.5, label='Hist.')
	plt.title('Time Satistics - Run {:d} - Events: {:d} - Total Time: {:.4e} s'.format(run, num_data, timeLimit))
	plt.xlabel('$\Delta$t (s)')
	plt.ylabel('Events')
	ax.grid(True)
	
	def exp_func(x, a, b, c, d):
		return (a*np.exp(-x/b) + c*np.exp(-x/d))
	center_bins = (bins[1:] + bins[:-1])/2
	popt, pcov = curve_fit(exp_func, center_bins, n, p0=(num_data/2, np.max(bins)/5, num_data/2, np.max(bins)/5))
	fitted_n = exp_func(center_bins, *popt)
	print('Mult. factor 1 = {:.4e}'.format(popt[0]))
	print('Time Const. 1 (s) = {:.4e}'.format(popt[1]))
	print('Mult. factor 2 = {:.4e}'.format(popt[2]))
	print('Time Const. 2 (s) = {:.4e}'.format(popt[3]))
	
	ax.plot(center_bins, fitted_n, '-b', linewidth=2, label='Double Exp. Fit')	
	ax.legend(loc='best')
	plt.show()

#Plot of the 'timeLimit' attributes
def time_limit(input_name, file_flag=True):
	if file_flag:
		f = h5py.File(input_name, 'r')
	else:
		f = input_name
	num_runs = len(f.keys()) - 1
	timeLimit_list = np.zeros(num_runs, dtype=np.float_)
	kmcSteps_list = np.zeros(num_runs, dtype=np.float_)
	run_list = np.arange(num_runs, dtype=np.int_)
	for run in run_list:
		dset_evo_name = 'run{:d}/evo'.format(run)
		timeLimit_list[run] = f[dset_evo_name].attrs['timeLimit']
		kmcSteps_list[run] = f[dset_evo_name].attrs['kmcSteps']
	print('Time Limit SUM: {:.4e} s'.format(np.sum(timeLimit_list)))
	try:
		dset_evo_name = 'merged/evo'.format(run)
		timeLimit_merged = f[dset_evo_name].attrs['timeLimit']
		print('Time Limit MERGED: {:.4e} s'.format(timeLimit_merged))
	except:
		print('No MERGED file...')
	if file_flag:
		f.close()
	
	#Power-of-10 markers on the axes 
	fmt = ticker.ScalarFormatter(useOffset=False)
	fmt.set_powerlimits((-2,2))
	
	fig = plt.figure(figsize=(16,10))
	ax1 = fig.add_subplot(1,2,1)
	ax1.plot(run_list, timeLimit_list, '-ob', linewidth=2)
	plt.title('Max Time - {:d} Runs'.format(num_runs))
	plt.xlabel('Run')
	plt.ylabel('Max Time (s)')
	ax1.grid(True)
	ax2= fig.add_subplot(1,2,2)
	ax2.yaxis.set_major_formatter(fmt)
	ax2.scatter(kmcSteps_list, timeLimit_list)
	plt.title('Max Time vs KMC Steps'.format(num_runs))
	plt.xlabel('KMC Steps')
	plt.ylabel('Max Time (s)')
	ax2.set_ylim(ax1.get_ylim())
	ax2.grid(True)
	plt.tight_layout()
	plt.show()