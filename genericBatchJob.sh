#!/bin/bash
#SBATCH --job-name=genBatch      ##Name of the job.
#SBATCH -A sbarwick_lab                  ##Account to charge to
#SBATCH -p standard                          ##Partition/queue name
#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum
#SBATCH --nodes=1                        ##Nodes to be used
#SBATCH --ntasks=1                       ##Numer of processes to be launched
#SBATCH --cpus-per-task=1                ##Cpu's to be used
#SBATCH --mem=18G

#SBATCH --output=genericBatchJob.out
#SBATCH --error=genericBatchJob.err

#SBATCH --mail-type=fail,end
#SBATCH --mail-user=rricesmi@uci.edu
export PYTHONPATH=$NuM:$PYTHONPATH
export PYTHONPATH=$Nu:$PYTHONPATH
export PYTHONPATH=$Radio:$PYTHONPATH
module load python/3.8.0
cd $ReflectiveAnalysis

python HRASimulation/HRAAnalysis.py
python HRASimulation/HRAAreaPlots.py
# python StationDataAnalysis/S02_StationDataAndDeeplearnTimeStrip.py
# python DeepLearning/SimpleSNR_Chi_plot.py
# python DeepLearning/D04B_reprocessNurPassingCut.py 30
# python DeepLearning/D01_convertSimNurToNpy.py
# python DeepLearning/D04C_CutInBacklobeRCR.py

# python CoreAnalysis/C04F_CalculateEffectiveAreaFromHdf.py ../CorePaperhdf5/gen2/*depth_300*
# python CoreAnalysis/C04E_corePaperNoiseDiagnostics.py 50 0.3
# python NeutrinoAnalysis/M01_generate_eventlist.py
# python DeepLearning/simpleTimeStrip.py
# python SimpleFootprintSimulation/Stn51RateCalc.py
# python CoreAnalysis/C04_corePaperDiagnostics.py 50 0.3


# python SimpleFootprintSimulatiom/Stn51Simulation.py SimpleFootprintSimulatiom/output/3.4.24_test.nur --min_energy 18.3 --max_energy 18.4
# python CoreAnalysis/C04_coreObjectAnalysis.py data/CoreDataObjects/CorePaper_CoreDataObjects_LPDA_2of4_100Hz_refl_300mRefl_SP_1R_0.3f_50.0dB_1.43781km_50000cores.pkl SP --title_comment LPDA_2of4_100MHz --savePrefix CorePaper/LPDA_2of4_100MHz
# python CoreAnalysis/C04_coreObjectAnalysis.py data/CoreDataObjects/CorePaper_CoreDataObjects_PA_8ch_100Hz_refl_300mRefl_SP_1R_0.3f_50.0dB_1.43781km_50000cores.pkl SP --title_comment PA_8ch_100Hz --savePrefix CorePaper/PA_8ch_100Hz

#python FootprintAnalysis/F01_FootprintSimulation.py 2 50 direct --type SP --config Stn51 --no_noise --add_amp --amp_type 300 --antenna lpda
#python FootprintAnalysis/F01_FootprintSimulation.py 2 50 direct --type MB --config MB_old --min_file 150 --no_noise --add_amp --amp_type 200 --antenna lpda

#python DeepLearning/D01_convertSimNurToNpy.py
#python DeepLearning/D02_trainCNN.py
#python DeepLearning/D03_probHist.py
# python DeepLearning/plotVisualExamples.py

#python CoreAnalysis/C00_coreAnalysisUtils.py
#python DeepLearning/testTimeStrip.py
#python A02CRrate.py

#python DeepLearning/D01_convertSimNurToNpy.py
#python DeepLearning/D0N_makeNeutrinoData.py

#python FootprintAnalysis/footprintIceRefraction.py 5 100 --dB 0 --depthLayer 300 --type IceTop --min_file 18.5 --max_file 18.6 --config Stn51 --num_icetop 20
#python FootprintAnalysis/S01_Generate_SP51_Events.py 5 10 --type IceTop --config Stn51 --add_amp
#python FootprintAnalysis/plotRefractedFootprints.py 7 --comment SP,Stn51,10k_cores,InfFirn,2/3_trigger -files FootprintAnalysis/data/CoREAS_direct_Stn51_wNoise_wAmp300s_InfAir_Layer300.0m_0.0dB_Area7.00_10000cores_part*
#python FootprintAnalysis/F01_FootprintSimulation.py 5 10 refracted --type MB --config MB_old --add_amp --amp_type 200
#python StationDataAnalysis/N02_SimulateNeutrinoEvents.py StationDataAnalysis/configs/MB_generic_100s_wDipole.json StationDataAnalysis/configs/reflectionConfigMoores.yaml StationDataAnalysis/data/N02_SimNu_100s_wNoise_wAmp_ StationDataAnalysis/data/N02_SimNu_100s_wNoise_wAmp_ -inputfilenames GeneratedNeutrinoEvents/*
#python FootprintAnalysis/F01_FootprintSimulation.py 5 100 refracted --type MB --config MB_old --add_amp --amp_type 100 --min_file 1500 --max_file 2000

#cd ../../../../pub/arianna/rricesmi/
#python DeepLearning/D01_convertSimNurToNpy.py
#python DeepLearning/D02_trainCNN.py
#python DeepLearning/D03_probHist.py


#python NeutrinoAnalysis/T01_generate_event_list.py

#SP all sigma
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/1e16_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_SP_footprint300m_infirn.json NeutrinoAnalysis/config.yaml NeutrinoAnalysis/output/SP_Allsigma_1e16_n1e5.hdf5 NeutrinoAnalysis/output/SP_Allsigma_1e16_n1e5.nur
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/1e17_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_SP_footprint300m_infirn.json NeutrinoAnalysis/config.yaml NeutrinoAnalysis/output/SP_Allsigma_1e17_n1e5.hdf5 NeutrinoAnalysis/output/SP_Allsigma_1e17_n1e5.nur
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/1e18_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_SP_footprint300m_infirn.json NeutrinoAnalysis/config.yaml NeutrinoAnalysis/output/SP_Allsigma_1e18_n1e5.hdf5 NeutrinoAnalysis/output/SP_Allsigma_1e18_n1e5.nur
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/1e19_n1e3.hdf5 NeutrinoAnalysis/station_configs/gen2_SP_footprint300m_infirn.json NeutrinoAnalysis/config.yaml NeutrinoAnalysis/output/SP_Allsigma_1e19_n1e3.hdf5 NeutrinoAnalysis/output/SP_Allsigma_1e19_n1e3.nur
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/1e20_n1e3.hdf5 NeutrinoAnalysis/station_configs/gen2_SP_footprint300m_infirn.json NeutrinoAnalysis/config.yaml NeutrinoAnalysis/output/SP_Allsigma_1e20_n1e3.hdf5 NeutrinoAnalysis/output/SP_Allsigma_1e20_n1e3.nur

#MB all sigma
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/MB_1e16_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_MB_infirn.json NeutrinoAnalysis/MB_config.yaml NeutrinoAnalysis/output/MB_Allsigma_1e16_n1e5.hdf5 NeutrinoAnalysis/output/MB_Allsigma_1e16_n1e5.nur --part 0009
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/MB_1e17_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_MB_infirn.json NeutrinoAnalysis/MB_config.yaml NeutrinoAnalysis/output/MB_Allsigma_1e17_n1e5.hdf5 NeutrinoAnalysis/output/MB_Allsigma_1e17_n1e5.nur --part 0009
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/MB_1e18_n1e5.hdf5 NeutrinoAnalysis/station_configs/gen2_MB_infirn.json NeutrinoAnalysis/MB_config.yaml NeutrinoAnalysis/output/MB_Allsigma_1e18_n1e5.hdf5 NeutrinoAnalysis/output/MB_Allsigma_1e18_n1e5.nur --part 0009

#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/MB_1e19_n1e3.hdf5 NeutrinoAnalysis/station_configs/gen2_MB_infirn.json NeutrinoAnalysis/MB_config.yaml NeutrinoAnalysis/output/MB_Allsigma_1e19_n1e3.hdf5 NeutrinoAnalysis/output/MB_Allsigma_1e19_n1e3.nur
#python NeutrinoAnalysis/T02_RunSimulation.py NeutrinoAnalysis/GeneratedEvents/MB_1e20_n1e3.hdf5 NeutrinoAnalysis/station_configs/gen2_MB_infirn.json NeutrinoAnalysis/MB_config.yaml NeutrinoAnalysis/output/MB_Allsigma_1e20_n1e3.hdf5 NeutrinoAnalysis/output/MB_Allsigma_1e20_n1e3.nur

