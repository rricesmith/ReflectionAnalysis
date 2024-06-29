#!/bin/bash
#SBATCH --job-name=CNN      ##Name of the job.
#SBATCH -A sbarwick_lab                  ##Account to charge to
#SBATCH -p standard                          ##Partition/queue name
#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum
#SBATCH --nodes=1                        ##Nodes to be used
#SBATCH --ntasks=1                       ##Numer of processes to be launched
#SBATCH --cpus-per-task=1                ##Cpu's to be used
#SBATCH --mem=18G

#SBATCH --output=DeepLearning/trainJob.out
#SBATCH --error=DeepLearning/trainJob.err

export PYTHONPATH=$NuM:$PYTHONPATH
export PYTHONPATH=$Nu:$PYTHONPATH
export PYTHONPATH=$Radio:$PYTHONPATH
module load python/3.8.0
cd $ReflectiveAnalysis

# python DeepLearning/D01_convertSimNurToNpy.py
#python DeepLearning/D02_trainCNN.py

#python DeepLearning/D03_processData.py 18
#python DeepLearning/D04_processAnalysis.py 13
python DeepLearning/D04B_reprocessNurPassingCut.py 30

#python DeepLearning/calcAverageNoiseFFT.py 13
#python DeepLearning/calcAverageNoiseFFT.py 15
#python DeepLearning/calcAverageNoiseFFT.py 18
