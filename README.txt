####
#	Code written by R. Rice-Smith
#	Available on 'github here'
####


###	S-Series






###	F-Series
F01_FootprintSimulation.py
	-CoREAS simulation that reads in Footprints and runs triggers depending upon station configurations
	-Configurations are Moores Bay (MB) current and future, and South Pole (SP) current and future

F02_createBatchJobs.py
	-Slurm job writer and executing function

F03_condenseFootprintPkl.py
	-Output of F01 is saved in multiple pkl files in a custom format
	-This file condenses multiple files ran simultaneously from F02 into a single one

