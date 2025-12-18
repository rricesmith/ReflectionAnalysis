
import numpy as np
import matplotlib.pyplot as plt
#import logging
import copy
import datetime
from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import NuRadioReco.framework.station
import NuRadioReco.framework.event
import NuRadioReco.framework.electric_field
import NuRadioReco.modules.channelResampler
#import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelLengthAdjuster
#import NuRadioReco.utilities.diodeSimulator
#import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.modules.base import module
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

import NuRadioReco.detector.antennapattern
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.detector import antennapattern

import NuRadioReco.modules.efieldToVoltageConverter
from NuRadioMC.SignalGen import parametrizations as signalgen
import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.modules.io.coreas import coreas

import astrotools.auger as auger
import pickle

#Can't use signalgen, use this?
from NuRadioMC.SignalGen import askaryan
import surfaceScatteringAreaEfficiency

from scipy.optimize import curve_fit

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

def plot_v_e_traces(trace, rad, A_scaling, B_scaling, n_samples):
    pulse = trace * A_scaling * rad ** (B_scaling - 1)
    antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
    antenna_pattern = antenna_pattern_provider.load_antenna_pattern('createLPDA_100MHz_InfFirn')
    samples = int(n_samples)
    freqs = np.fft.rfftfreq(samples, 1./n_samples)
    signal_zenith = 90 * units.deg
    signal_azimuth = 0 * units.deg
    antenna_response = antenna_pattern.get_antenna_response_vectorized(freqs, signal_zenith, signal_azimuth, 180.*units.deg, 0., 90*units.deg, 180*units.deg) #check 0's and deg values
    theta = antenna_response['theta']                                                                           
#    print(f'len pulse {len(test_pulse_sc)} len freqs {len(freqs)} len antenna response {len(theta)}')                                                                     
#    print(f'fft time2freq len {len(fft.time2freq(test_pulse_sc, n_samples))}')                                                                                            
    channel_spectrum = antenna_response['theta'] * fft.time2freq(pulse, n_samples)                                                                                        
    channel_v_trace = fft.freq2time(channel_spectrum, n_samples)                                                                                                          
#    print(f'len final trace {len(channel_v_trace)}')                                                                                                                     

    plt.plot(channel_v_trace, label='v_trace')                                                                                                                                             
    plt.legend()
    # plt.show()
    plt.savefig('CoreAnalysis/plots/HorPropSurfaceAskaryanVTraceExample.png')
    plt.plot(pulse, label='e_trace')
#    plt.scatter(np.arange(0, len(channel_v_trace), 1), channel_v_trace, label='v_trace')                                                                                 
#            plt.scatter(np.arange(0, len(channel_v_trace), 1), test_pulse_sc, label='e_trace')                                                                           
    plt.legend()                                                                                                                                                          
#            plt.yscale('log')                                                                                                                                            
    # plt.show()         
    plt.savefig('CoreAnalysis/plots/HorPropSurfaceAskaryanETraceExample.png')                                                                                                                                                   
    return                                        

#det = detector.Detector(json_filename='configurations/gen2_4LPDA_PA_15m_RNOG_300k_200mdipole_infirn.json')
det = detector.Detector(json_filename='configurations/single_surface_4LPDA_PA_15m_RNOG_300k.json')
det.update(datetime.datetime(2018,1,1))

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin(number_of_samples=400, offset=50)

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin()

use_LPDA = False

event = NuRadioReco.framework.event.Event(1,1) #What is this doing?
station = NuRadioReco.framework.station.Station(1) #Why 101? Change? Needs to be same as json file probably, set to 1?

lpda_channel = NuRadioReco.framework.channel.Channel(0) #Need to have an array? Maybe can just do triggering off single antenna for now
dip_channel = NuRadioReco.framework.channel.Channel(9)

antenna_provider = antennapattern.AntennaPatternProvider()
LPDA = antenna_provider.load_antenna_pattern('createLPDA_100MHz_InfFirn')
dipole = antenna_provider.load_antenna_pattern('RNOG_vpol_v1_n1.4')

if use_LPDA:
    use_channels = [0]
else:
    use_channels = [9]

lpda_electricField = NuRadioReco.framework.electric_field.ElectricField([0])
dip_electricField = NuRadioReco.framework.electric_field.ElectricField([9])

type = 'SP'
threshold = 26
if type == 'MB':
    atm_overburden = 1000
    f = 0.06
elif type == 'SP':
    atm_overburden = 680
    f = 0.3


#Askaryan pulse parameters, update!!!!!!!!!!!!!!!!!!!!1
energy = 1e16 * units.eV
fhad = 0.5 #?
viewing_angle = 55.8 * units.deg
n_samples = 2**12 #?
dt = 0.5 * units.ns
n_index = 1.78
R = 1 * units.m


rad = np.linspace(1, 2000, 2000)
#energies = np.linspace(1, 100, 100)
#energies = np.logspace(0.0, 3.0, num=50)
#energies = [100.]
energies_edges = (10 ** np.arange(16, 20.01, 0.1)) / energy
energies = 0.5 * (energies_edges[1:] + energies_edges[:-1])

#equations = [[44, -1.4, 4, 180., 0., True], [44, -1.4, 4, 180., 90., True], [44, -1.4, 4, 90., 0., True], [44, -1.4, 4, 90., 90., True], [44, -1.4, 4, 0., 0., True], [44, -1.4, 4, 0., 90., True]]
#equations = [[44, -1.4, 4, 180., 0., False], [44, -1.4, 4, 180., 90., False], [44, -1.4, 4, 90., 0., False], [44, -1.4, 4, 90., 90., False], [44, -1.4, 4, 0., 0., False], [44, -1.4, 4, 0., 90., False]]
#equations = [[44, -1.4, 4, 180., 0., False], [44, -1.4, 4, 0., 0., False]]
equations = [[44, -1.4, 4, 90., 90., True]] 

#fig, ax = plt.subplots(len(equations))


iter_r = 0
max_r = np.ones((len(equations),len(energies))) * min(rad)
for A, B, sigma, inc_zen, inc_az, theta_bool in equations:
    azimuth = inc_az*units.deg
    iter = 0
    for i_E in energies:

        print(f'Now doing energy {i_E}')

        n_steps = 20
        prev_r = 0
        r = 100

#        for r in rad:
        while n_steps != 0:
            test_pulse_sc = askaryan.get_time_trace(energy * i_E, viewing_angle, n_samples, dt, shower_type='had', n_index=n_index, R=1 * units.m, model='Alvarez2009')

            zero_trace = np.array([np.zeros(len(test_pulse_sc)), np.zeros(len(test_pulse_sc)),np.zeros(len(test_pulse_sc))])
            if theta_bool:
                pulse_trace = np.array([np.zeros(len(test_pulse_sc)), test_pulse_sc * A * r ** (B-1), np.zeros(len(test_pulse_sc))])
#                electricField.set_trace(np.array([np.zeros(len(test_pulse_sc)), test_pulse_sc * A * r ** (B-1), np.zeros(len(test_pulse_sc))]), 10 * units.GHz)                                   #theta polarized
            else:
                pulse_trace = np.array([np.zeros(len(test_pulse_sc)), np.zeros(len(test_pulse_sc)), test_pulse_sc * A * r ** (B-1)])
#                electricField.set_trace(np.array([np.zeros(len(test_pulse_sc)), np.zeros(len(test_pulse_sc)), test_pulse_sc * A * r ** (B-1)]), 10 * units.GHz) #phi polarized

            if use_LPDA:
                lpda_electricField.set_trace(pulse_trace, 10*units.GHz)
                dip_electricField.set_trace(zero_trace, 10*units.GHz)
            else:
                lpda_electricField.set_trace(zero_trace, 10*units.GHz)
                dip_electricField.set_trace(pulse_trace, 10*units.GHz)

#            electricField.set_parameter(efp.polarization_angle, 0*units.deg) #0 is all etheta, 90 all ephi

            station.set_is_cosmic_ray()		#Don't seem to need this anymore, should include?

#Old way, nor sure it works properly            
#            voltage_trace = trace_utilities.get_channel_voltage_from_efield(station, electricField, use_channels, det, inc_zen, azimuth, antenna_provider, return_spectrum=False)

            station.remove_triggers()


#Copy code from efieldtovoltageconverter method

            time_resolution = dt
            channel_spectrum = None

            station.set_electric_fields([lpda_electricField, dip_electricField])
            for electric_field in station.get_electric_fields_for_channels(use_channels):

#                if use_LPDA:
#                lpda_channel = NuRadioReco.framework.channel.Channel(0)
#                else:
#                dip_channel = NuRadioReco.framework.channel.Channel(9)
                new_efield = NuRadioReco.framework.base_trace.BaseTrace()
                new_efield.set_trace(electric_field.get_trace(), electric_field.get_sampling_rate())
                trace_length_samples = int(round(electric_field.get_number_of_samples() / time_resolution))
#                trace_length_samples = int(round(4000/dt))
                new_trace = np.zeros((3, trace_length_samples))

                start_time = electric_field.get_trace_start_time()
                start_bin = int(round(start_time / time_resolution))
                new_trace[:, start_bin:(start_bin + new_efield.get_number_of_samples())] = new_efield.get_trace()

                trace_object = NuRadioReco.framework.base_trace.BaseTrace()
                trace_object.set_trace(new_trace, 1. / time_resolution)

                ff = trace_object.get_frequencies()
                efield_fft = trace_object.get_frequency_spectrum() 
#                VEL = trace_utilities.get_efield_antenna_factor(station, ff, [0], det, inc_zen, azimuth, antenna_provider)
                if use_LPDA:
                    lpda_VEL = LPDA.get_antenna_response_vectorized(ff, np.deg2rad(inc_zen), np.deg2rad(azimuth), np.deg2rad(180), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90))
#                    lpda_VEL = LPDA.get_antenna_response_vectorized(ff, inc_zen, azimuth, 180.*units.deg, 0.*units.deg, 90.*units.deg, 90.*units.deg)
                else:
                    dip_VEL = dipole.get_antenna_response_vectorized(ff, np.deg2rad(inc_zen), np.deg2rad(azimuth), np.deg2rad(0), np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
#                    dip_VEL = dipole.get_antenna_response_vectorized(ff, inc_zen, azimuth, 0.*units.deg, 0.*units.deg, 90.*units.deg, 0.*units.deg)
#                VEL = VEL[0] #selectring for one channel only					#this this was get_efield_antenna_factor method
#                voltage_fft = np.sum(VEL * np.array([efield_fft[1], efield_fft[2]]), axis=0)
                if use_LPDA:
                    VEL = lpda_VEL
                else:
                    VEL = dip_VEL

                voltage_fft = np.array(VEL['theta'] * efield_fft[1] + VEL['phi'] * efield_fft[2])


                if(channel_spectrum is None):
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft

#                if trace_object is None:
                if use_LPDA:
                    lpda_channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())
                    dip_channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())
#                    dip_channel.set_trace(np.zeros(trace_length_samples), trace_object.get_sampling_rate())
#                    print('is none?')
                else:
                    lpda_channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())
                    dip_channel.set_frequency_spectrum(channel_spectrum, trace_object.get_sampling_rate())
#                    lpda_channel.set_trace(np.zeros(trace_length_samples), trace_object.get_sampling_rate())
#                print(f'num chans {station.get_number_of_channels()}')

#                print('we plottin')


                station.add_channel(lpda_channel)
                station.add_channel(dip_channel)
#                looper += 1
#                break
#                print(f'num chans 2 {station.get_number_of_channels()}')
#                quit()
            

#            print(f'loop {looper}')
#            station.set_sim_station(sim_station)

#            print(f'has sim station? {station.has_sim_station()}')
#            eventTypeIdentifier.run(event, station, 'forced', 'cosmic_ray')
#            efieldToVoltageConverter.run(event, station, det)
 

            channelBandPassFilter.run(
                event,
                station,
                det,
                passband = [1 * units.MHz, 0.15], 
                filter_type = 'butter',
                order=10,
                rp=0.1
            )
            channelBandPassFilter.run(
                event,
                station,
                det,
                passband = [0.08, 800 * units.GHz],
                filter_type = 'butter',
                order=5,
                rp=0.1
            )


            channelResampler.run(event, station, det, sampling_rate=1 * units.GHz)
            channelSignalReconstructor.run(event, station, det)
            channelLengthAdjuster.run(event, station, det)                     #Is this needed?

            trigger_name = 'trigger_' + str(i_E) + '_' + str(r) + '_' + str(inc_zen) + '_' + str(inc_az) + '_' + str(theta_bool)
            print('trigger name ' +trigger_name)
        
#LPDA Trigger
            if use_LPDA:
                thresh = 1.3662817627550142e-05 / 3.94  #LPDA threshold
            else:
                 thresh = threshold * units.micro * units.V
#                thresh = 1.689172534356015e-05          #Dipole threshold

            triggerSimulator.run(
                event,
                station,
                det,
                threshold_high = thresh * sigma * units.V,
                threshold_low = -thresh * sigma * units.V,
                high_low_window = 20 * units.ns,
                coinc_window = 32 * units.ns,
                number_concidences = 1,
                triggered_channels = use_channels,
                trigger_name = trigger_name
            )

            """
            if station.has_triggered(trigger_name=trigger_name):
                max_r[iter_r, iter] = r
            else:
                
                type='LPDA'
                if not use_LPDA:
                    type='Dipole'
                pol='theta'
                if not theta_bool:
                    pol='phi'
                print('lets go')
                ax[iter_r].plot(channel.get_trace())
                ax[iter_r].set_title(type + ' ' + pol + ' zen ' + str(inc_zen) + ' azi ' + str(inc_az))
#                plt.show()
                print('plotted...')
                

                break
            """
            if station.has_triggered(trigger_name=trigger_name):
                prev_r = r
                if max_r[iter_r, iter] < r:
                    max_r[iter_r, iter] = r
                r = r * 2
            else:
                step_size = (r - prev_r) / 2
                r = r - step_size
            n_steps = n_steps - 1


#            print('no trigger, continue')
#        print(f'n_trigger is {n_trigger}')
        iter += 1
    iter_r += 1

"""
fig.legend()
fig.suptitle('Voltage trace for ' + str(i_E) + ' * 10**17eV')
plt.show()
"""



#plt.figure()
iter_r = 0
ees = energy*energies
for A, B, sigma, inc_zen, inc_az, theta_bool in equations:
# 180.*units.deg, 0.*units.deg, 90.*units.deg, 90.*units.deg LPDA config
# 0.*units.deg, 0.*units.deg, 0.*units.deg, 90.*units.deg dipol config
    if theta_bool:
        pol = 'theta'
    else:
        pol = 'phi'
    if use_LPDA:
        plt.scatter(np.log10(ees), max_r[iter_r], label=pol + ' pol, inc zen '+str(inc_zen) + ', inc azi ' + str(inc_az))
    else:
        plt.scatter(np.log10(ees), max_r[iter_r], label=pol + ' pol, inc zen '+str(inc_zen) + ', inc azi ' + str(inc_az))


    iter_r += 1

fit_energies = np.log10(ees)
soln, cov =  curve_fit(func_powerlaw, fit_energies, max_r[0], maxfev=5000)
print(f'soln {soln}')

plt.plot(fit_energies, func_powerlaw(fit_energies, *soln), '--', label=f'{soln[1]}x^{soln[0]}+{soln[2]}')
plt.plot(fit_energies, func_powerlaw(fit_energies, 18.4797, 6.8*10**-22, 5.06), label='aprox fit')

plt.ylabel('Max Radius Triggered (m)')
plt.xlabel('Energy (eV)')
if use_LPDA:
    atype = 'LPDA'
else:
    atype = 'dipole'
plt.title(f'Max Radius Triggered Surface Scattering, {type} {atype} {thresh}muV')
#plt.xscale('log')
#plt.legend(['44*r^-1.4'])
plt.legend()
# plt.show()
plt.savefig(f'CoreAnalysis/plots/HorPropSurfaceAskaryanMaxRadiusTriggered_{type}_{atype}_{thresh}muV.png')
plt.clf()


d_eng = 0.5	#Half step up and down per energy in log10
e_rate_engs = np.log10(ees)

hp_evt_rate_mb = np.zeros(len(e_rate_engs))
hp_evt_rate_sp = np.zeros_like(e_rate_engs)

for iE, eng in enumerate(e_rate_engs):
    Aeff = (max_r[0] * 10**-3) ** 2









spacing = [500, 750, 1000, 1500]
#spacing = [500]
frac_cov = np.zeros((len(spacing), len(energies)))
event_rate = np.zeros((len(spacing), len(energies)))
Aeff = np.zeros((len(spacing), len(energies)))
eng_bins_half_width = (energies[1] - energies[0]) / 2


"""
print(max_r)
print(max_r[0])
for i_S in range(len(spacing)):
    for i_f in range(len(frac_cov[i_S])):
#        print(f'spacing {spacing[i_S]} rad {max_r[0][i_f]} gives cov { surfaceScatteringAreaEfficiency.area_covered(spacing[i_S], max_r[0][i_f])}')
        frac_cov[i_S][i_f] = surfaceScatteringAreaEfficiency.area_covered(spacing[i_S], max_r[0][i_f])
        tri_area = surfaceScatteringAreaEfficiency.area_equilateral_triangle(spacing[i_S] * 10 ** -3)
        eLow = energy * (energies[i_f] - eng_bins_half_width)
        eHigh = energy * (energies[i_f] + eng_bins_half_width)
        print(f'low {eLow} high {eHigh} log low {np.log10(eLow)} log high {np.log10(eHigh)}')
        event_rate[i_S][i_f] = auger.event_rate(np.log10(eLow), np.log10(eHigh), 85, tri_area) * frac_cov[i_S][i_f]
        print(f'event rate {auger.event_rate(np.log10(eLow), np.log10(eHigh), 85, tri_area)}')
        print(f'frac cov {frac_cov[i_S][i_f]} tri area {tri_area} eLow {np.log10(eLow)} eHigh {np.log10(eHigh)} event rate {event_rate[i_S][i_f]}')
        Aeff[i_S][i_f] = tri_area * frac_cov[i_S][i_f]

#    plt.scatter(energy * energies, frac_cov[i_S], label='Spacing ' + str(spacing[i_S]) + 'm')
#    plt.scatter(energy * energies, event_rate[i_S], label='Spacing ' + str(spacing[i_S]) + 'm')
    plt.scatter(energy * energies, Aeff[i_S], label='Spacing ' + str(spacing[i_S]) + 'm')

plt.legend()
plt.xlabel('Energy eV')
plt.xscale('log')
plt.ylabel('Aeff (km^2)')
plt.show()
"""


flux = 'auger_19'
with open(f"data/output_{flux}_{atm_overburden}.pkl", "rb") as fin:
    shower_energies, weights_shower_energies, shower_xmax = pickle.load(fin)

dCos = 0.05
coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
n_zen_bins = len(coszen_bin_edges)-1

shower_E_bins = np.arange(17, 20.01, 0.1)
logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])

log_E_r_edges = np.log10(energies_edges * energy)
log_E_r = (log_E_r_edges[1:] + log_E_r_edges[:-1]) / 2

Aeff = np.zeros((len(spacing), len(log_E_r)))
tri_area = np.zeros(len(spacing))
event_rate_cr_eng = np.zeros((len(spacing), len(logEs)))
event_rate_surface_shower = np.zeros_like(event_rate_cr_eng)
total_rate_cr = np.zeros_like(event_rate_cr_eng)
total_rate_surface = np.zeros_like(event_rate_cr_eng)

for iS, space in enumerate(spacing):
    tri_area[iS] = surfaceScatteringAreaEfficiency.area_equilateral_triangle(space * 10**-3)
    for iE, logEr in enumerate(log_E_r):
        frac_cov = surfaceScatteringAreaEfficiency.area_covered(space, max_r[0][iE])
#        Aeff[iS][iE] = tri_area[iS] * frac_cov
        Aeff[iS][iE] = np.pi * (max_r[0][iE] * 10**-3) ** 2


for iS, space in enumerate(spacing):
    for iE, logE in enumerate(logEs):
        for iC, coszen in enumerate(coszens):
            mask = ~np.isnan(shower_energies[iE][iC])
            engiEiC_org = np.log10(shower_energies[iE][iC][mask])
            engiEiC = np.log10(shower_energies[iE][iC][mask] * f)
            weightsiEiC = weights_shower_energies[iE][iC][mask]
            surf_shower_digit = np.digitize(engiEiC, log_E_r_edges)-1
            rate_digit = np.digitize(engiEiC_org, shower_E_bins) - 1
            for n in range(len(engiEiC)):
                engN = surf_shower_digit[n]
                rateN = rate_digit[n]
                if engN < 0:
                    continue
                event_rate_cr_eng[iS][iE] += weightsiEiC[n] * Aeff[iS][engN]
                if rateN < 0:
                    continue                
                event_rate_surface_shower[iS][rateN] += weightsiEiC[n] * Aeff[iS][engN]
#                total_rate_cr[iS][iE] += weightsiEiC[n] * tri_area[iS]
#                print(f'original energy of {engiEiC_org[n]} between {shower_E_bins[rateN]} and {shower_E_bins[rateN+1]}')
#                print(f'reduced energy of {engiEiC[n]} between {log_E_r_edges[engN]} and {log_E_r_edges[engN+1]}')
#                total_rate_surface[iS][rateN] += weightsiEiC[n] * tri_area[iS]

reduced_logEs = []
for iE, logE in enumerate(logEs):
    if logE >= 16:
        reduced_logEs.append(logE)

red_e_r_cr_eng = np.zeros( (len(spacing), len(reduced_logEs)) )
red_e_r_surf_sh = np.zeros_like(red_e_r_cr_eng)
for iS, space in enumerate(spacing):
    i = 0
    for iE, logE in enumerate(logEs):
        if logE >= 16:
#        if True:
            red_e_r_cr_eng[iS][i] = event_rate_cr_eng[iS][iE]
            red_e_r_surf_sh[iS][i] = event_rate_surface_shower[iS][iE]
            i += 1

logEs = reduced_logEs
event_rate_cr_eng = red_e_r_cr_eng
event_rate_surface_shower = red_e_r_surf_sh

for iS, space in enumerate(spacing):
    plt.scatter(logEs, event_rate_cr_eng[iS], label=f'{space}m, {sum(event_rate_cr_eng[iS]):.6f}evts/yr/stn')
plt.title(f'Event Rate of Core Scattering per Original CR Energy, {type} f={f} {threshold}muV')
plt.ylabel('Events/year/station')
plt.xlabel('Cosmic Ray Energy (log10 eV)')
plt.yscale('log')
plt.legend()
# plt.show()
plt.savefig(f'CoreAnalysis/plots/HorPropSurfaceAskaryanEventRateCR_{type}_f{f}_{threshold}muV.png')
plt.clf()
for iS, space in enumerate(spacing):
    plt.scatter(logEs, event_rate_surface_shower[iS], label=f'{space}m, {sum(event_rate_surface_shower[iS]):.6f}evts/yr/stn')
plt.title(f'Event Rate of Core Scattering per Core Energy Remaining, {type} f={f} {threshold}muV')
plt.ylabel('Events/year/station')
plt.xlabel('Core Energy at Surface (log10 eV)')
plt.yscale('log')
plt.legend()
# plt.show()
plt.savefig(f'CoreAnalysis/plots/HorPropSurfaceAskaryanEventRateSurface_{type}_f{f}_{threshold}muV.png')
plt.clf()   
for iS, space in enumerate(spacing):
    plt.scatter(logEs + np.log10(f), event_rate_surface_shower[iS], label=f'{space}m, {sum(event_rate_surface_shower[iS]):.6f}evts/yr/stn')
plt.title(f'Event Rates of Core Scattering per Coupled Core energy, {type}, f={f} {threshold}muV')
plt.ylabel('Events/year/station')
plt.xlabel('Coupled Core Energy (log10eV)')
plt.yscale('log')
plt.legend()
# plt.show()
plt.savefig(f'CoreAnalysis/plots/HorPropSurfaceAskaryanEventRateCoupledCore_{type}_f{f}_{threshold}muV.png')
plt.clf()


"""
Old methods when doing multiple spacing plots for a triangle configuration of stations

cr_frac = event_rate_cr_eng / total_rate_cr
for iS, space in enumerate(spacing):
    plt.scatter(logEs, cr_frac[iS], label='Spacing ' +str(space) + 'm')
plt.title(f'Fraction of Core Scatters Detected per Original CR Energy, {type} f={f} {threshold}muV')
plt.ylabel('% Cores Detected')
plt.xlabel('Cosmic Ray Energy (log10 eV)')
;#plt.yscale('log')
plt.legend()
plt.show()
core_frac = event_rate_surface_shower / total_rate_surface
for iS, space in enumerate(spacing):
    plt.scatter(logEs, core_frac[iS], label='Spacing ' +str(space) + 'm')
plt.title(f'Fraction of Core Scatters Detected per Core Energy Remaining, {type} f={f} {threshold}muV')
plt.ylabel('% Cores Detected')
plt.xlabel('Core Energy at Surface (log10 eV)')
#plt.yscale('log')
plt.legend()
plt.show()

cr_missed = total_rate_cr - event_rate_cr_eng
for iS, space in enumerate(spacing):
    plt.scatter(logEs, cr_missed[iS], label='Spacing ' +str(space) + f'm, {sum(cr_missed[iS]):.2f}evts/yr/stn')
plt.title(f'Event Rate of Undetected Cores as Original CR Energy, {type} f={f} {threshold}muV')
plt.ylabel('Events/year/station')
plt.xlabel('Cosmic Ray Energy (log10 eV)')
plt.yscale('log')
plt.legend()
plt.show()
core_missed = total_rate_surface - event_rate_surface_shower
for iS, space in enumerate(spacing):
    plt.scatter(logEs, core_missed[iS], label='Spacing ' +str(space) + f'm, {sum(core_missed[iS]):.2f}evts/yr/stn')
plt.title(f'Event Rate of Undetected Cores as Core Energy Remaining, {type} f={f} {threshold}muV')
plt.ylabel('Events/year/station')
plt.xlabel('Core Energy at Surface (log10 eV)')
plt.yscale('log')
plt.legend()
plt.show()
"""



"""
total_rate = np.zeros_like(event_rate)

for iS, space in enumerate(spacing):
    for iE, eng in enumerate(energies):
        frac_cov[iS][iE] = surfaceScatteringAreaEfficiency.area_covered(space, max_r[0][iE])
        tri_area = surfaceScatteringAreaEfficiency.area_equilateral_triangle(space * 10 ** -3)
        Aeff[iS][iE] = tri_area * frac_cov[iS][iE]
        for iC, coszen in enumerate(coszens):
            yy = shower_energies[:, iC].flatten()
            mask = ~np.isnan(yy)
            yy = yy[mask]
            ww = weights_shower_energies[:, iC].flatten()[mask]
            N, bin_edges = np.histogram(np.log10(yy), bins=shower_E_bins, weights=ww)
            event_rate[iS][iE] += Aeff[iS][iE] * N[iE]
            total_rate[iS][iE] += tri_area * N[iE]

    plt.scatter(energy * energies, event_rate[iS], label='Ntrig, Spacing ' + str(spacing[iS]) + 'm')
    plt.scatter(energy * energies, total_rate[iS], label='Ntotal, Spacing ' + str(spacing[iS]) + 'm')



#plt.title('Efficiency per spacing assuming Dipole theta polarized 90deg signal antenna response')
plt.title('Events per year detected inside triangular station configuration of Cores Striking the Surface')
#plt.title('Effective Area per Energy of triangular staton configuration')
plt.legend()
plt.xscale('log')
plt.yscale('log')
#plt.ylabel('% Triangle Detectable')
plt.ylabel('Events per year per station')
#plt.ylabel('Aeff (km^2)')
plt.xlabel('Energy (eV)')
plt.show()


scatter_trig_frac = event_rate / total_rate
for iS, space in enumerate(spacing):
    plt.scatter(energy * energies, scatter_trig_frac[iS], label='Spacing ' + str(space) + f'm')

plt.title('Fraction of Core Scatters Detected at Moores Bay')
plt.legend()
plt.xscale('log')
plt.ylabel('% Cores Detected')
plt.xlabel('Energy (eV)')
plt.show()

scatter_untrig_rate = total_rate - event_rate
for iS, space in enumerate(spacing):
    plt.scatter(energy * energies, scatter_untrig_rate[iS], label='Spacing ' + str(space) + f'm, {sum(scatter_untrig_rate[iS]:0f}')

plt.title('Cores Not Detected at Moores Bay')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Events/year/station')
plt.xlabel('Energy (eV)')
plt.show()
"""
