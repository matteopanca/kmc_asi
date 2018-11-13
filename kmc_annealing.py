import kmcasi as kmc
import numpy as np
import h5py
import sys
import time

start_time = time.perf_counter()
# start_time = time.process_time()

#Import the file list
with open(sys.argv[1],'r') as f:
	file_list = f.readlines()

for i in range(len(file_list)):
	print(file_list[i].rstrip())
	#Import the configuration file
	control_list = []
	with open(file_list[i].rstrip(),'r') as f:
		control_list.append(f.readline().rstrip()) #File with DOUBLE Probs.
		control_list.append(f.readline().rstrip()) #File with SINGLE Probs.
		control_list.append(float(f.readline().rstrip())) #Attempt frequency - Double (Hz)
		control_list.append(float(f.readline().rstrip())) #Attempt frequency - Single (Hz)
		control_list.append(int(f.readline().rstrip())) #Vertices in each row
		control_list.append(int(f.readline().rstrip())) #Vertices in each column
		control_list.append(f.readline().rstrip()) #Init type ('1', '2', '3', '4' or 'r')
		control_list.append(f.readline().rstrip()) #Boundary conditions: 'finite' (default) or 'pbc'
		control_list.append(float(f.readline().rstrip())) #Disorder (in fractional units)
		control_list.append(int(f.readline().rstrip())) #Steps for each run
		control_list.append(float(f.readline().rstrip())) #Time limit for each run
		control_list.append(int(f.readline().rstrip())) #Number of images to be saved
		num_runs = int(f.readline().rstrip()) #Number of runs
		output_name = f.readline().rstrip() #Output file
	prob_double = np.loadtxt(control_list[0], dtype=np.float_)
	prob_single = np.loadtxt(control_list[1], dtype=np.float_)

	#Running the simulation (ignoring num_runs)
	if i == 0:
		output_file = h5py.File(output_name, 'w')
		previousObj = kmc.Array(control_list, (prob_double, prob_single))
		previousObj.run(i)
		kmc.save_evolution(output_file, previousObj)
	else:
		obj = kmc.Array(previousObj, control_list, (prob_double, prob_single))
		obj.run(i)
		kmc.save_evolution(output_file, obj)
		previousObj = obj #it hasn't to be a copy...
kmc.merge_runs(output_file)
output_file.close()

stop_time = time.perf_counter()
# stop_time = time.process_time()
print('Elapsed time (s): {:.4f}'.format(stop_time - start_time))