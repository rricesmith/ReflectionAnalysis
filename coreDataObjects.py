import numpy as np
import matplotlib.pyplot as plt


def coreStatistics(core_bin_lower_energy, cores_max=5000, cores_min=1000, DEBUG=False):
    #Pass in the lower energy of the bin being simulated, along with any changes core statistics
    #Returns the number of cores used for that simulation bin
    negative_scaling = 1600
    cores = cores_max - negative_scaling * (core_bin_lower_energy - 14)
    cores = int(cores)
    if DEBUG:
        print(f'first test, with energy {core_bin_lower_energy} going to use cores {cores}')
    if cores < cores_min:
        return cores_min
    else:
        return int(cores)

def throwsPerRadBin(spacing, rad_bins, n_throws, shape='square'):
    if shape=='square':
        area = spacing**2
    elif shape=='circle':
        area = np.pi * spacing**2
    else:
        print(f'Wrong input shape error, no {shape} shape')
        quit()
    throwsPerArea = n_throws / area
    rad_areas = np.zeros(len(rad_bins)-1)
    for iR in range(len(rad_areas)):
        rad_areas[iR] = np.pi * (rad_bins[iR+1]**2 - rad_bins[iR]**2)
    print(f'throws per area {throwsPerArea} and rad areas {rad_areas}')
    return rad_areas * throwsPerArea

def weightedError(sigma_values, DEBUG=False):
    #Sigma bins should be an array of the error per bin
    sigmas = np.array(sigma_values)
    weights = 1 / sigmas**2
    weights[sigmas == 0] = 0
    sum = np.sum(weights)
    denominator = np.sqrt(sum)
    if DEBUG:
        print(f'testing weighted error')
        print(f'sigmas {sigmas}')
        print(f'weights {weights}')
        print(f'sum {sum}')
        print(f'denominator {denominator}')
        print(f'result average error {1 / np.sqrt(np.sum(weights))}')
    if denominator == 0:
        return 0
    #Multiply by number of non-zero bins who contibuted error to get total error of all bins combined
    n_bins = np.count_nonzero(sigma_values)
    return n_bins * 1 / denominator

def weightedAverage(x_values, sigma_values, DEBUG=False):
    x = np.array(x_values)
    sigma = np.array(sigma_values)
    weights = 1 / sigma**2
    weights[sigma == 0] = 0
    numerator = np.sum(weights * x)
    denominator = np.sum(weights)
    if DEBUG:
        print(f'testing weighted average')
        print(f'x {x}')
        print(f'sigmas {sigma}')
        print(f'weights {weights}')
        print(f'num {numerator}, denom {denominator}, result {numerator / denominator}')
    if denominator == 0:
        return 0
    #Multiply by number of non-zero bins who contibuted error to get total error of all bins combined
    n_bins = np.count_nonzero(sigma_values)
    return n_bins * numerator / denominator

def f_FromFitEcoreEcr(Ecore, Ecr):
    #Ecore and Ecr need to be in non-log scale
    if Ecore < 30:
        #Only time its so low is because values are log10eV
        Ecore = 10**Ecore
    if Ecr < 30:
        Ecr = 10**Ecr
    ratio = Ecore/Ecr
    #Using best fit line from Simon's work, Gen2 Radio call 28/3/22
    slope = 0.568
    intercept = -0.0871

    f = slope * ratio + intercept
    if f < 0:
        f = 0
    return f

def f_FromFitXmax(Xmax, zenith, Xice=680):
    #Units of depth are in g/cm^2
    #680 g/cm^2 is SP
    #1000 g/cm^2 is Sea Level (MB)
    #Zenith should be in degrees
     
    #Best fit from Simon's work, Gen2 Radio Call 28/3/22
    # exponential, y=Ae^-Bx + C    

    A = 0.242
    B = -0.00808
    C = 0.0429

    #Get slant depth from surface (linear distance left to propagate to surface)
    slant = (Xice - Xmax) / np.cos(np.deg2rad(zenith))

    f = A * np.exp(B * slant) + C
    return f


"""
class aeffBin:

    def __init__(self, E_low, E_high, coszen_low, coszen_high,
                rr_trig_hist, rad_bins, max_radius, n_cores, shape='square'):
        
        self.e_bins = [E_low, E_high]
        self.e_center = (E_low + E_high)/2
        self.coszen_bins = [coszen_low, coszen_high]
        self.coszen_bins = np.sort(self.coszen_bins)
        self.coszen_center = (coszen_low + coszen_high)/2
        self.zen_center = np.arccos(self.coszen_center)

        self.radius_trig_hist = rr_trig_hist
        self.rad_bins = rad_bins
        self.max_radius = max_radius    #In units km
        self.n_cores = n_cores
        self.shape = shape

        self.setAeff(rad_bins, rr_trig_hist)


    def setAeff(self, rad_bins, rr_trig_hist):
        #Sets the Aeff per rad bin, as well as the Aeff error
        #Error is +- sqrt(n_trig) in bin
        #So value becomes area * (n_trig +- sqrt(n_trig))/n_throws
        self.rad_bins = rad_bins
        areas = np.zeros(len(rr_trig_hist))
        for iR in range(len(areas)):
            areas[iR] = np.pi * (rad_bins[iR+1]**2 - rad_bins[iR]**2) * 10**-6
        if len(rad_bins) == 0:
            #Case were there are no triggers, set all Aeff to zero
            self.Aeff_per_rad_bin = 0 * areas
#            self.Aeff_error_high_rad_bin = 0 * areas
#            self.Aeff_error_low_rad_bin = 0 * areas
            self.Aeff_error_per_rad_bin = 0 * areas
            self.weightedAverageAeff = weightedAverage(self.Aeff_per_rad_bin, self.Aeff_error_per_rad_bin)
            self.weightedErrorAeff = weightedError(self.Aeff_error_per_rad_bin)
            return
        self.throwsPerRadBin = throwsPerRadBin(self.max_radius * 10**3, rad_bins, self.n_cores, self.shape)
        
        reflTrigFrac = rr_trig_hist / self.throwsPerRadBin
        reflTrigFrac[reflTrigFrac > 1] = 1

        print(f'is trig hist the problem?')
        print(f'rr trig hist {rr_trig_hist}')
        print(f'reflTrigFrac is {reflTrigFrac}')
        error_hist = np.sqrt(rr_trig_hist)
        print(f'error hist {error_hist}')
        error_per_bin = error_hist / self.throwsPerRadBin
        error_per_bin[error_per_bin > 1] = 1
        print(f'error per bin {error_per_bin}')
#        errorTrigHigh = (rr_trig_hist + error_hist) / self.throwsPerRadBin
#        errorTrigLow = (rr_trig_hist - error_hist) / self.throwsPerRadBin
#        errorTrigLow[errorTrigLow < 0] = 0

        self.statistics_per_bin = rr_trig_hist
        self.Aeff_per_rad_bin = areas * reflTrigFrac
        self.Aeff_error_per_rad_bin = areas * error_per_bin
        print(f'Aeff per rad bin {self.Aeff_per_rad_bin}')
        print(f'Aeff error per rad bin {self.Aeff_error_per_rad_bin}')
        self.weightedAverageAeff = weightedAverage(self.Aeff_per_rad_bin, self.Aeff_error_per_rad_bin)
        self.weightedErrorAeff = weightedError(self.Aeff_error_per_rad_bin)
        print(f'sum Aeff {sum(self.Aeff_per_rad_bin)} sum error {sum(self.Aeff_error_per_rad_bin)}')
        print(f'weight Aeff {self.weightedAverageAeff} weight error {self.weightedErrorAeff}')
        if self.weightedErrorAeff > sum(self.Aeff_per_rad_bin):
            if abs(self.weightedErrorAeff - sum(self.Aeff_per_rad_bin))/sum(self.Aeff_per_rad_bin) < 0.01:
                print(f'Error slightly larger than Aeff, ignoring')
            else:
                print(f'Error larger than the sum Aeff, bugfix needed')
                quit()
#        self.Aeff_error_high_rad_bin = areas * errorTrigHigh
#        self.Aeff_error_low_rad_bin = areas * errorTrigLow
"""




class coresBin:
    rad_bins = []
    Aeff_per_rad_bin = []
    trigRatesPerRadBin = []
#    parentCRs = []


    def __init__(self, E_low, E_high, coszen_low, coszen_high, 
                rr_trig_hist, rad_bins, max_radius, n_cores, shape='square', 
                zeniths = [], viewing_angles = [], weights = [], 
                arrival_zen = [], SNR = []):
    ####
    #Rad Bins should be in units m
    ####
        self.parentCRs_TA = []
        self.parentCRs_Auger = []
        self.e_bins = [E_low, E_high]
        self.e_center = (E_low + E_high)/2
        self.coszen_bins = [coszen_low, coszen_high]
        self.coszen_bins = np.sort(self.coszen_bins)
        self.coszen_center = (coszen_low + coszen_high)/2
        self.zen_center = np.arccos(self.coszen_center)

         #Old internal Aeff
        self.radius_trig_hist = rr_trig_hist
        self.rad_bins = rad_bins
        self.max_radius = max_radius    #In units km
        self.n_cores = n_cores
        self.shape = shape
        self.zenBins = np.linspace(0, np.pi, 30)
        self.zenHist, bins = np.histogram(zeniths, self.zenBins)
        self.viewing_angles = viewing_angles
        self.weights = weights
        self.arrival_zen = arrival_zen
        self.SNR = SNR

        self.setAeff(rad_bins, rr_trig_hist)
        self.setSingleAeff(max_radius, rr_trig_hist, n_cores)
#        self.setAeff(aeffObjectList, n_cores)   #n_cores stands for number of Xmax simmed per CR. So 2000 in current tests

    def setZeniths(self, zeniths):
        self.zenBins = np.linspace(0, np.pi, 30)
        self.zenHist, edges = np.histogram(zeniths, self.zenBins)
        
    def getZeniths(self):
        try:
            return self.zenHist
        except AttributeError:
            self.setZeniths([])
            return self.zenHist

    def addCrParent(self, E_cr, Xmax, eRatePerArea, type='TA'):
        parentCR = crEvent(E_cr, Xmax, eRatePerArea)
        if type == 'TA':
            self.parentCRs_TA.append(parentCR)
        elif type == 'Auger':
            self.parentCRs_Auger.append(parentCR)
        else:
            print(f'Type {type} unsupported')
            quit()
    """
    def addCrParent(self, crParent):
        self.parentCRs.append(crParent)
    """

    def setSingleAeff(self, max_radius, rr_trig_hist, n_cores):
        #Sets the Aeff for total area of simulation
        #Aeff is in km^2
        n_trig = sum(rr_trig_hist)
        area = np.pi * (max_radius) ** 2
        self.singleAeff = area * n_trig / n_cores
        self.singleErrorAeff = (area * (n_trig + np.sqrt(n_trig)) / n_cores) - self.singleAeff
        print(f'DEBUG')
        print(f'n_trig {n_trig} from rr trig hist {rr_trig_hist} for cores {n_cores}')
        print(f'total area {area} from max rad {max_radius}')
        print(f'single Aeff resultant {self.singleAeff}')
        print(f'single error Aeff {self.singleErrorAeff}')

         #OLD setAeff, before doing variable f
    def setAeff(self, rad_bins, rr_trig_hist):
        #Sets the Aeff per rad bin, as well as the Aeff error
        #Error is +- sqrt(n_trig) in bin
        #So value becomes area * (n_trig +- sqrt(n_trig))/n_throws
        self.rad_bins = rad_bins    #In units m
        areas = np.zeros(len(rr_trig_hist))     #Will end up being units km^2
        for iR in range(len(areas)):
            areas[iR] = np.pi * (rad_bins[iR+1]**2 - rad_bins[iR]**2) * 10**-6
        if len(rad_bins) == 0:
            #Case were there are no triggers, set all Aeff to zero
            self.Aeff_per_rad_bin = 0 * areas
#            self.Aeff_error_high_rad_bin = 0 * areas
#            self.Aeff_error_low_rad_bin = 0 * areas
            self.Aeff_error_per_rad_bin = 0 * areas
            self.weightedAverageAeff = weightedAverage(self.Aeff_per_rad_bin, self.Aeff_error_per_rad_bin)
            self.weightedErrorAeff = weightedError(self.Aeff_error_per_rad_bin)
            return
        self.throwsPerRadBin = throwsPerRadBin(self.max_radius * 10**3, rad_bins, self.n_cores, self.shape)
        
        reflTrigFrac = rr_trig_hist / self.throwsPerRadBin
        reflTrigFrac[reflTrigFrac > 1] = 1

#        print(f'is trig hist the problem?')
#        print(f'rr trig hist {rr_trig_hist}')
#        print(f'reflTrigFrac is {reflTrigFrac}')
        error_hist = np.sqrt(rr_trig_hist)
#        print(f'error hist {error_hist}')
        error_per_bin = error_hist / self.throwsPerRadBin
        error_per_bin[error_per_bin > 1] = 1
#        print(f'error per bin {error_per_bin}')
#        errorTrigHigh = (rr_trig_hist + error_hist) / self.throwsPerRadBin
#        errorTrigLow = (rr_trig_hist - error_hist) / self.throwsPerRadBin
#        errorTrigLow[errorTrigLow < 0] = 0

        self.statistics_per_bin = rr_trig_hist
        self.Aeff_per_rad_bin = areas * reflTrigFrac
        self.Aeff_error_per_rad_bin = areas * error_per_bin
        print(f'Aeff per rad bin {self.Aeff_per_rad_bin}')
        print(f'Aeff error per rad bin {self.Aeff_error_per_rad_bin}')
        self.weightedAverageAeff = weightedAverage(self.Aeff_per_rad_bin, self.Aeff_error_per_rad_bin)  #Units km^2
        self.weightedErrorAeff = weightedError(self.Aeff_error_per_rad_bin)
        print(f'sum Aeff {sum(self.Aeff_per_rad_bin)} sum error {sum(self.Aeff_error_per_rad_bin)}')
        print(f'weight Aeff {self.weightedAverageAeff} weight error {self.weightedErrorAeff}')
        if self.weightedErrorAeff > sum(self.Aeff_per_rad_bin):
            if abs(self.weightedErrorAeff - sum(self.Aeff_per_rad_bin))/sum(self.Aeff_per_rad_bin) < 0.01:
                print(f'Error slightly larger than Aeff, ignoring')
            else:
                print(f'Error larger than the sum Aeff, bugfix needed')
                quit()
#        self.Aeff_error_high_rad_bin = areas * errorTrigHigh
#        self.Aeff_error_low_rad_bin = areas * errorTrigLow
    """
    def setAeff(self, aeffObjectList, n_cores):
        #Will iterate through parent CRs, map the core energy to Aeff through the CR 'f', and add that Aeff to this structures Aeff
        # aeffObjectList - list of Aeff objects as in this file
        # n_cores - Number of cores simulated per Ecr. This means the the Aeff added per CR is reduced by 1/n_cores

        ####Scratch that, above is wrong
        #Calculating a single Aeff doesn't make sense
    """

    def totalTrigRateCore(self, singleAeff=False, type='TA'):
        try:
            if type == 'TA':
                return self.totalTrigEventRate_TA
            elif type == 'Auger':
                return self.totalTrigEventRate_Auger
            else:
                print(f'Type {type} not supported')
                quit()
        except AttributeError:
            #Set TA rate
            totalEventRate = 0
            totalSingleEventRate = 0
            parentCRs = self.parentCRs_TA
            for CR in parentCRs:
                totalSingleEventRate += self.singleAeff * CR.eRatePerArea
                totalEventRate += sum(self.Aeff_per_rad_bin) * CR.eRatePerArea
#                totalEventRate += self.weightedAverageAeff * CR.eRatePerArea
            self.totalTrigEventRate_TA = totalEventRate
            self.totalTrigSingleEventRate_TA = totalSingleEventRate

            #Set Auger rate
            totalEventRate = 0
            totalSingleEventRate = 0
            parentCRs = self.parentCRs_Auger
            for CR in parentCRs:
                totalSingleEventRate += self.singleAeff * CR.eRatePerArea
                totalEventRate += sum(self.Aeff_per_rad_bin) * CR.eRatePerArea
#                totalEventRate += self.weightedAverageAeff * CR.eRatePerArea
            self.totalTrigEventRate_Auger = totalEventRate
            self.totalTrigSingleEventRate_Auger = totalSingleEventRate

            if type == 'TA':
                if singleAeff:
                    return self.totalTrigSingleEventRate_TA
                return self.totalTrigEventRate_TA
            elif type == 'Auger':
                if singleAeff:
                    return self.totalTrigSingleEventRate_Auger
                return self.totalTrigEventRate_Auger
            else:
                print(f'Type {type} not supported')
                quit()

    def totalEventRateCore(self, singleAeff=False, type='TA'):
#        print(f'tot aeff core {self.totalAeffCore()}')
#        print(f'total erate area core {self.totalEventRateAreaCore()}')
        if singleAeff:
            return self.singleAeff * self.totalEventRateAreaCore(type=type)
        return self.totalAeffCore() * self.totalEventRateAreaCore(type=type)

    def totalEventErrorCore(self, singleAeff=False, highLow=True, type='TA'):
        #old way I did error, wrong I believe because each bin is independent, so want to have the sqrt of the sum of the squares
#        aeffErrorHigh = sum(self.Aeff_error_high_rad_bin)
#        aeffErrorLow = sum(self.Aeff_error_low_rad_bin)
#        eHigh = np.array(self.Aeff_error_high_rad_bin)
#        eLow = np.array(self.Aeff_error_low_rad_bin)
#        aeffErrorHigh = np.sqrt(np.sum(eHigh**2))
#        aeffErrorLow = np.sqrt(np.sum(eLow**2))

        #Want to do a weighted average below, ch7 of Taylor intro to error analysis
#        errorHigh = self.weightedAverageAeff + self.weightedErrorAeff
#        print(f'weighted avg aeff {self.weightedAverageAeff} and error {self.weightedErrorAeff}')
#        errorLow = self.weightedAverageAeff - self.weightedErrorAeff
#        print(f'error Low is {errorLow}')
        if highLow == False:
            return self.weightedErrorAeff * self.totalEventRateAreaCore(type=type)
        errorHigh = self.totalAeffCore() + self.weightedErrorAeff
        errorLow = self.totalAeffCore() - self.weightedErrorAeff
        if errorLow < 0:
            errorLow = 0
        return errorHigh * self.totalEventRateAreaCore(type=type), errorLow * self.totalEventRateAreaCore(type=type)
#        return self.totalEventRateAreaCore() * aeffErrorHigh, self.totalEventRateAreaCore() * aeffErrorLow


    def totalEventRateAreaCore(self, type='TA'):
        try:
            if type == 'TA':
                return self.totalEventRateArea_TA
            elif type == 'Auger':
                return self.totalEventRateArea_Auger
            else:
                print(f'Type {type} not supported')
                quit()
        except AttributeError:
            totalEventRate = 0
            parentCRs = self.parentCRs_TA
            for CR in parentCRs:
                totalEventRate += CR.eRatePerArea
            self.totalEventRateArea_TA = totalEventRate

            totalEventRate = 0
            parentCRs = self.parentCRs_Auger
            for CR in parentCRs:
                totalEventRate += CR.eRatePerArea
            self.totalEventRateArea_Auger = totalEventRate

            if type == 'TA':
                return self.totalEventRateArea_TA
            elif type == 'Auger':
                return self.totalEventRateArea_Auger
            else:
                print(f'Type {type} not supported')
                quit()

    def totalAeffCore(self):
        return sum(self.Aeff_per_rad_bin)
#        return self.weightedAverageAeff

    def setTotalEventRatePerArea(self, type='TA'):
        totalEventRatePerArea = 0
        parentCRs = self.parentCRs_TA
        for CR in parentCRs:
            totalEventRatePerArea += CR.eRatePerArea
        self.totalEventRatePerArea_TA = totalEventRatePerArea

        totalEventRatePerArea = 0
        parentCRs = self.parentCRs_Auger
        for CR in parentCRs:
            totalEventRatePerArea += CR.eRatePerArea
        self.totalEventRatePerArea_Auger = totalEventRatePerArea

        if type == 'TA':
            return self.totalEventRatePerArea_TA
        elif type == 'Auger':
            return self.totalEventRatePerArea_Auger
        else:
            print(f'Type {type} not supported')
            quit()

    def eventRatePerRad(self, type='TA'):
        if not len(self.rad_bins) > 0:
            return [], []
        eventRatePerRad = np.zeros(len(self.rad_bins)-1)
        if type == 'TA':
            parentCRs = self.parentCRs_TA
        elif type == 'Auger':
            parentCRs = self.parentCRs_Auger
        else:
            print(f'Type {type} not supported')
            quit()
        for CR in parentCRs:
            eventRatePerRad += self.Aeff_per_rad_bin * CR.eRatePerArea
        return eventRatePerRad, self.rad_bins

    def eventRatePerRadPerEcr(self, EcrLow = 0, EcrHigh = 21, type='TA', return_Xmax=False):
        E_crs = []
        eRates = []
        errorRate = []
        Xmaxs = []
#        errorRateHigh = []
#        errorRateLow = []
        if type == 'TA':
            parentCRs = self.parentCRs_TA
        elif type == 'Auger':
            parentCRs = self.parentCRs_Auger
        else:
            print(f'Type {type} not supported')
            quit()
        for CR in parentCRs:
            E_crs.append(CR.E_cr)
            Xmaxs.append(CR.Xmax)
            eRates.append(self.Aeff_per_rad_bin * CR.eRatePerArea)
#            errorRate.append(self.Aeff_error_per_rad_bin * CR.eRatePerArea)
            errorRate.append(self.weightedErrorAeff * CR.eRatePerArea)
 #           errorRateHigh.append(self.Aeff_error_high_rad_bin * CR.eRatePerArea)
 #           errorRateLow.append(self.Aeff_error_low_rad_bin * CR.eRatePerArea)
        if return_Xmax:
            return E_crs, Xmaxs, eRates, errorRate, self.rad_bins
        return E_crs, eRates, errorRate, self.rad_bins
        

    def eventRatePerEcr(self, eLow = 0, eHigh = 21, type=type, return_Xmax=False):
        if return_Xmax:
            E_crs, Xmaxs, eRates, errorRate, rad_bins = self.eventRatePerRadPerEcr(EcrLow=eLow, EcrHigh=eHigh, type=type, return_Xmax=return_Xmax)
        else:
            E_crs, eRates, errorRate, rad_bins = self.eventRatePerRadPerEcr(EcrLow=eLow, EcrHigh=eHigh, type=type)
        totErates = np.zeros_like(E_crs)
        totError = np.zeros_like(E_crs)
#        totErrorHigh = np.zeros_like(E_crs)
#        totErrorLow = np.zeros_like(E_crs)
        """
        for iR, rates in enumerate(eRates):
            totErates[iR] = np.sum(rates)
        #changing error sum to be sqrt of sum of squares, since errors are independent per bin
        for iR, rates in enumerate(errorRateHigh):
#            rate = np.array(rates)
#            totErrorHigh[iR] = np.sqrt( np.sum( rate**2))
            totErrorHigh[iR] = np.sum(rates)
        for iR, rates in enumerate(errorRateLow):
#            rate = np.arrat(rates)
#            totErrorLow[iR] = np.sqrt( np.sum( rate**2))
            totErrorLow[iR] = np.sum(rates)
        """
        #Need to do weighted average, as below
        for iR, rates in enumerate(eRates):
#            totErates[iR] = weightedAverage(rates, errorRate[iR])
            totErates[iR] = np.sum(rates)
#            totError[iR] = weightedError(errorRate[iR])

#        return E_crs, totErates, totError
        if return_Xmax:
                return E_crs, Xmaxs, totErates, errorRate
        return E_crs, totErates, errorRate

    def plotEventRatePerRad(self, station=[0,0], n_lines=400, type='TA'):
        if not self.hasParents() or not (len(self.rad_bins) > 0):
            return False
        eventRatePerArea = self.totalEventRatePerArea()
        if not eventRatePerArea:
            return False
#        eventRatePerArea = self.eventRatePerArea()
#        if (sum(self.trigRatesPerRadBin)==0) or (sum(eventRatePerRad) == 0):
#            print(f'No trigs in this bin')
#            return False
        xx = np.linspace(-1.1 * self.rad_bins[-1], 1.1 * self.rad_bins[-1], num=n_lines)
        yy = np.linspace(-1.1 * self.rad_bins[-1], 1.1 * self.rad_bins[-1], num=n_lines)
        contourVals = np.zeros( (n_lines, n_lines) )
        area_per_bin = (xx[0] - xx[1]) * (yy[0] - yy[1]) * 10**-3
        eRatePerRad,  = self.eventRatePerRad(type=type)
        for iX, x in enumerate(xx):
            for iY, y in enumerate(yy):
                r = np.sqrt(x**2 + y**2)
                r_dig = np.digitize(r, self.rad_bins) - 1
                if r_dig >= (len(self.rad_bins)-1):
                    continue
#                print(f'event rate of {area_per_bin * self.trigRatesPerRadBin[r_dig] * eventRatePerRad[r_dig]} from {area_per_bin} * {self.trigRatesPerRadBin[r_dig]} * {eventRatePerRad[r_dig]}')
                contourVals[iX][iY] = area_per_bin * self.trigRatesPerRadBin[r_dig] * eRatePerRad[r_dig]

        xx += station[0]
        yy += station[1]
        plt.contourf(xx, yy, contourVals, level=10, cmap='YlOrRd')
#        plt.contour(xx, yy, contourVals, cmap='YlOrRd')
#        plt.colorbar(label='Events/Stn/Yr')
        return True

    def addEventsToArea(self, xx, yy, zz, type='TA'):
    ###
    #Takes in an array of x and y bins (in meters) of area triggered over
    #zz is just an empty array that will become the event rate inside that area
    ###
        if not self.hasParents() or not (len(self.rad_bins) > 0):
            return xx, yy, zz
#        eventRatePerArea = self.totalEventRatePerArea()
        totEventRateArea = self.setTotalEventRatePerArea(type)
#        print(f'event rate per area {self.totalEventRatePerArea}')
#        print(f'saved valued {self.totalEventRatePerArea}')
        if totEventRateArea == 0:
            return xx, yy, zz
        if (sum(self.trigRatesPerRadBin)==0):
            return xx, yy, zz
        areaPerBin = (xx[0] - xx[1]) * (yy[0] - yy[1]) * 10**-6
#        print(f'Looking at trig rates {self.trigRatesPerRadBin} over bins {self.rad_bins} with area {areaPerBin}')
        for iX, x in enumerate(xx):
            for iY, y in enumerate(yy):
                r = np.sqrt(x**2 + y**2)
                r_dig = np.digitize(r, self.rad_bins)
                if r_dig >= (len(self.rad_bins)-1):
                    continue
                zz[iX][iY] += areaPerBin * self.trigRatesPerRadBin[r_dig] * totEventRateArea
        return xx, yy, zz


    def engInBins(self, eCore):
        return self.e_bins[0] <= eCore < self.e_bins[1]

    def coszenInBins(self, coszen):
        return self.coszen_bins[0] <= coszen < self.coszen_bins[1]

    def zenInBins(self, zenith, deg=True):
        zen_bins = np.sort( np.arccos(self.coszen_bins) )
        zen_bins[np.isnan(zen_bins)] = 0
        if deg:
            zen_bins = np.rad2deg(zen_bins)
        print(f'Checking if {zen_bins[0]} <= {zenith} < {zen_bins[1]} of value {zen_bins[0] <= zenith < zen_bins[1]}')
        return zen_bins[0] <= zenith < zen_bins[1]

    def hasParents(self, type='TA'):
        if type == 'TA' and not self.parentCRs_TA:
            return False
        elif type == 'Auger' and not self.parentCRs_Auger:
            return False
        return True


class crEvent:

#    def __init__(self, E_cr, Xmax, zenith, eRatePerArea, loc='SP'):
        #Zenith should be in degrees
    def __init__(self, E_cr, Xmax, eRatePerArea):
        self.E_cr = E_cr
        self.eRatePerArea = eRatePerArea
        self.Xmax = Xmax
#        self.zenith = zenith

#        if loc == 'SP':
#            self.f = f_FromFitXmax(Xmax, zenith, Xice=680)
#        elif loc == 'MB':
#            self.f = f_FromFitXmax(Xmax, zenith, Xice=1000)
