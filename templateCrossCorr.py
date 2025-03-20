import numpy as np
import fractions
from decimal import Decimal
from scipy import signal
from radiotools import helper as hp
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import pickle

from scipy.signal import correlate

def match_sampling(ref_template, resampling_factor):
    if(resampling_factor.numerator != 1):
        ref_template_resampled = signal.resample(ref_template, resampling_factor.numerator * len(ref_template))
    else:
        ref_template_resampled = ref_template
    if(resampling_factor.denominator != 1):
        ref_template_resampled = signal.resample(ref_template_resampled, int(len(ref_template_resampled) / resampling_factor.denominator))
    return ref_template_resampled



#method derived from NuRadioReco/modules/channelTemplateCorrelation

def get_xcorr_for_channel(orig_trace, template_trace, orig_sampling_rate, template_sampling_rate, times=[], debug=False, SNR='n/a'):
    if len(orig_trace) == 0:
        return 0
    orig_trace = orig_trace / max(np.abs(orig_trace))
    template_trace = template_trace / max(np.abs(template_trace))

    orig_binning = 1. / template_sampling_rate
    target_binning = 1. / orig_sampling_rate
    resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning))
#    print(f'resampling factor {resampling_factor}')

    ref_template_resampled = match_sampling(template_trace, resampling_factor)
    orig_max = np.argmax(np.abs(orig_trace))
#    plt.plot(times, orig_trace, label='orig trace')
#    half_len = int(len(ref_template_resampled)/2)
#    if half_len < len(orig_trace)/2:				#If our template trace is shorter than our data, we cut around our data
#        cut_start = orig_max - half_len
#        cut_end = orig_max + half_len
    full_len = int(len(ref_template_resampled))
    if full_len < len(orig_trace):
        cut_start = orig_max - 50
        cut_end = orig_max - 50 + full_len
        if cut_start < 0:
            cut_end += np.abs(cut_start)
            cut_start += np.abs(cut_start)
        if debug:
            times = times[cut_start:cut_end]      
        orig_trace = orig_trace[cut_start:cut_end]
#    plt.plot(times, orig_trace, label='cut trace')
#    plt.legend()
#    plt.show()

    ref_template_resampled = ref_template_resampled / max(ref_template_resampled)
#    print(f'orig trace {orig_trace} and ref temp {ref_template_resampled}')
    xcorr_trace = hp.get_normalized_xcorr(orig_trace, ref_template_resampled)
#    xcorr_trace = correlate(orig_trace, ref_template_resampled)

    xcorrpos = np.argmax(np.abs(xcorr_trace))
    xcorr = xcorr_trace[xcorrpos]

    if debug:
        print(f'xcorr of {xcorr}')
        plot_trace = orig_trace / max(orig_trace)
        if len(times) == 0:
            plt.plot(plot_trace, label='Measured')
            plt.plot(hp.get_normalized_xcorr(orig_trace, ref_template_resampled), label='Template')
        else:
            flip = np.sign(xcorr)
#            plt.plot(times, orig_trace, label='Measured')
#            plt.plot(times - min(times),orig_trace, label='Measured')
            argmax = np.argmax(orig_trace)
            dt = times[1]-times[0]
            tttemp = np.arange(0, len(ref_template_resampled)*dt, dt)
            print(f'len times {len(times)} and template {len(ref_template_resampled)}')
            print(f'len of temp {len(ref_template_resampled)}, xcorrpos {xcorrpos}')
            print(f'len of orig trace {len(orig_trace)}')
#            plt.plot(times, orig_trace, label='Measured')
            plt.plot(tttemp, orig_trace, label='Measured')
#            plt.plot(tttemp, ref_template_resampled * np.abs(orig_trace).max(), '--', label='Template')
            plt.plot(tttemp, flip * np.roll(ref_template_resampled * np.abs(orig_trace).max(), xcorrpos), '--', label='Template')
#            plt.xlim(times[argmax] - 150 * units.ns, times[argmax] + 150*units.ns)
#        plt.plot(np.abs(orig_trace).max() * xcorr_trace / max(xcorr_trace), label='xcorr trace')
#        plt.plot(xcorr_trace, label='xcorr trace')
        plt.xlabel('ns')
        plt.legend()
        plt.title(f'Xcorr {xcorr}, {SNR}SNR')
        plt.show()

    """
    data_dump = {}
    data_dump['19log10eV,66zen,107azi'] = orig_trace       
    plt.plot(orig_trace)
    plt.show()
    with open('reflectedCR_template.pkl', 'wb') as output:
        pickle.dump(data_dump, output)
    output.close()

    quit()
    """

    return xcorr

