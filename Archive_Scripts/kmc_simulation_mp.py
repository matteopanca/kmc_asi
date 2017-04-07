#THIS DOESN'T WORK: problem with h5py...

import multiprocessing as mp
import kmc_package as kmc
import numpy as np
import h5py
import sys
import time

def target_fun(myObj, i):
	myObj.run(i)

if __name__ == '__main__':
	
	start_time = time.perf_counter()
	# start_time = time.process_time()
	
	#Import the configuration file
	control_list = []
	with open(sys.argv[1],'r') as f:
		control_list.append(f.readline().rstrip()) #File with DOUBLE Probs.
		control_list.append(f.readline().rstrip()) #File with SINGLE Probs.
		control_list.append(float(f.readline().rstrip())) #Attempt frequency (Hz)
		control_list.append(int(f.readline().rstrip())) #Vertices in each row
		control_list.append(int(f.readline().rstrip())) #Vertices in each column
		control_list.append(f.readline().rstrip()) #Init type ('1', '2', '3', '4' or 'r')
		control_list.append(f.readline().rstrip()) #Boundary conditions: 'finite' (default) or 'pbc'
		control_list.append(float(f.readline().rstrip())) #Steps for each run
		control_list.append(int(f.readline().rstrip())) #Number of images to be saved
		num_runs = int(f.readline().rstrip()) #Number of runs
		output_name = f.readline().rstrip() #Output file
	prob_double = np.loadtxt(control_list[0], dtype=np.float_)
	prob_single = np.loadtxt(control_list[1], dtype=np.float_)
	
	output_file = h5py.File(output_name, 'w')
	#Running the simulation (num_runs times)
	arrayObj = [kmc.Array(control_list, (prob_double, prob_single), output_file) for i in range(num_runs)]
	jobs = [] #Put the processes in a list
	for i in range(num_runs):
		p = mp.Process(target=target_fun, args=(arrayObj[i], i))
		jobs.append(p)
	#Start all processes and wait each process to be finished
	for p in jobs:
		p.start() #Start
	for p in jobs:
		p.join() #Wait
	kmc.avg_runs(output_file)
	output_file.close()
	
	stop_time = time.perf_counter()
	# stop_time = time.process_time()
	print('Elapsed time (s): {:.4f}'.format(stop_time - start_time))