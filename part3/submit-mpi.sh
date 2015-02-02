#!/bin/bash
	# @ job_name		= heatmpi
	# @ partition		= debug
	# @ initialdir		= .
	# @ output		= heatmpi.%j.out
	# @ error		= heatmpi.%j.err
	# @ total_tasks		= 8
	# @ cpus_per_task	= 1
	# @ tasks_per_node	= 8
	# @ wall_clock_limit	= 00:02:00

prog=heatmpi

procs=1
mpirun -n $procs ./$prog test.dat

procs=2
mpirun -n $procs ./$prog test.dat

procs=4
mpirun -n $procs ./$prog test.dat

procs=8
mpirun -n $procs ./$prog test.dat

