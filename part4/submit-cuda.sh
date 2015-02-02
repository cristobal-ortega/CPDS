#!/bin/bash
	# @ job_name		= heatCUDA
	# @ partition		= debug
	# @ initialdir		= .
	# @ output		= heatCUDA.%j.out
	# @ error		= heatCUDA.%j.err
	# @ total_tasks		= 1
	# @ gpus_per_node	= 1
	# @ wall_clock_limit	= 00:02:00

prog=./heatCUDA_residual
txb=16

${prog} test.dat -t $txb

