from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from NuRadioReco.modules.base.module import register_run
import h5py
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from radiotools import coordinatesystems as cstrafo
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
import numpy as np
import numpy.random
import logging
import time
import os
from icecream import ic
from NuRadioReco.framework.parameters import electricFieldParameters as efp

import matplotlib.pyplot as plt

class readCoREAS:
    """
    coreas input module for fixed grid of stations.
    This module distributes core positions randomly within a user defined area and calculates the electric field
    at the detector positions as specified in the detector description by choosing the closest antenna
    of the star shape pattern simulation
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_files = None
        self.__station_id = None
        self.__n_cores = None
        self.__max_distace = None
        self.__current_input_file = None
        self.__random_generator = None
        self.logger = logging.getLogger('NuRadioReco.readCoREAS')

    def begin(self, input_files, xmin, xmax, ymin, ymax, n_cores=10, shape='square', seed=None, log_level=logging.INFO):
        """
        begin method

        initialize readCoREAS module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        xmin: float
            minimum x coordinate of the area in which core positions are distributed in meters
        xmax: float
            maximum x coordinate of the area in which core positions are distributed in meters
        ymin: float
            minimum y coordinate of the area in which core positions are distributed in meters
        ynax: float
            maximum y coordinate of the area in which core positions are distributed in meters
        n_cores: number of cores (integer)
            the number of random core positions to generate for each input file
        shape: string
            shape of area in which core positions are distributed, 'square', 'triangle', 'uniform', or 'linear' configured currently
            if triangle, area is considered to be area with vertices (xmin, ymin), (xmax, ymin), (0.5(xmax-xmin), ymax)
            linear produces cores in a line along (xmin, ymin) to (xmax, ymax) spacred by n_cores
            uniform produces an evenly spaced rectangular grid of throws
            radial produces a distribution spaced evenly in sqrt(radius) over a circle centered on (xmin, ymin) (xmax, ymax), with diameter=xmax-xmin, leading to equal throws per radius
        seed: int (default: None)
            Seed for the random number generation. If None is passed, no seed is set
        """
        self.__input_files = input_files
        self.__n_cores = n_cores
        self.__current_input_file = 0
        self.__shape = shape
        self.__area = [xmin, xmax, ymin, ymax]

        self.__random_generator = numpy.random.RandomState(seed)
        self.logger.setLevel(log_level)

    def modify_eField(self, electric_field, ant_surface_position, ant_ice_position, cr_xmax, ray_type, refl_layer_depth = 0*units.m, 
                            reflective_dB=0, force_dB = False, attenuation_model=None):
        """
        Take in a NuRadioReco Electric Field object
        And modify the trace of the electric field by 1/R travel distance,
        By reflectivity if using a reflective layer,
        And modify the zenith at the antenna to reflect the ice-refraction and reflection.

        Parameters
        ----------
        electric_field: Electric Field object
            Electric Field object whose trace is to be modified
        ant_surface_position: [x, y, z] floats
            Calculated surface position of antenna based on zenith and depth in meters
        ant_ice_position: [x, y, z] floats
            Position of the antenna in ice as listed in the detector description in meters
        cr_xmax: float
            Xmax of the cosmic ray in (what units?)
        ray_type: string
            'refracted' for modification of direct signals to deep antennas
            'reflected' for modification of reflected signals
            'by_depth' for modification of signals based on if they are above or below a reflective layer
        refl_layer_depth: positive float
            If using a reflective layer, depth of the reflective layer in meters
        reflective_dB: float
            If using reflective layer, relative dB loss in layer
        force_dB: boolean
            If True, modifies electric fields by dB power loss even if ray type is refraction. No effect if ray type is reflective
        attenuation_model: string (default None)
            Attenuation model to be used.
            'MB_flat' applies a flat attenuation length of 460m - 180m/GHz * frequency
        """        
        n_ice = 1.78

        # Removing zenith adjustment, if station is in cosmic_ray mode then it will be computed when convoluting antenna response
        refr_zenith = np.arcsin( np.sin(electric_field[efp.zenith]) / n_ice) * units.rad

        # Untested configuration
        if ray_type == 'reflected':
            # DEPRECIATED? Remove possible
            ant_ice_position[2] = 2 * refl_layer_depth + ant_ice_position[2]
            refr_zenith = np.pi - refr_zenith
        ray = np.array(ant_surface_position) - np.array(ant_ice_position)
        dist_traveled = np.sqrt(ray[0]**2 + ray[1]**2 + ray[2]**2)
#        print(f'refr zen {refr_zenith}, ray zenith is {np.arctan(ray[2]/np.sqrt(ray[0]**2+ray[1]**2) )}')
#        print(f'ant surface position {ant_surface_position} and ice position {ant_ice_position} creates ray {ray} for dist of {dist_traveled}')

#        efield_adjust = cr_xmax / (cr_xmax + dist_traveled)
#        print(f'efield adjust from cr xmax {efield_adjust} due to crxmax {cr_xmax} and dist traveled {dist_traveled}')
        if attenuation_model == None:
            efield_adjust = 1
        elif attenuation_model == 'MB_flat':
            att_length = 420*units.m
            efield_adjust = np.exp(-np.abs(dist_traveled) / att_length)
        elif attenuation_model == 'MB_freq':
#            freqs = electric_field.get_frequencies() * units.GHz
            freqs = electric_field.get_frequencies()
            att_length = 460 * units.m - 180 * units.m / units.GHz * freqs / units.GHz
            att_length[att_length <= 0] = 1
            efield_adjust = np.exp(-dist_traveled / (att_length/units.m) )
#            print(f'efield adjust from atten {np.exp(-dist_traveled / att_length)} of dist {dist_traveled}')
            efield_adjust[att_length <= 0] = 0	#Remove negative attenuation lenghts
            efield_adjust[efield_adjust > 1] = 1 #Remove increases in electric field

#            plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[0]), label='Pre-att 0')
#            plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[1]), label='Pre-att 1')
#            plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[2]), label='Pre-att 2')

        else:
            print(f'There exists no attenuation model {attenuation_model}')
            quit()

        """
        pre_val = 0
        for iF, fre in enumerate(freqs):
#            print(f'our fre is {fre/units.MHz} and has truth value {190 <= fre/units.MHz <= 200}')
            if 190 <= fre/units.MHz <= 200:
                for i in range(20):
                    pre_val += np.abs(electric_field.get_frequency_spectrum()[2])[iF+10-i]               
                break 
        """

        """
        plt.plot(freqs/units.MHz,efield_adjust)
        plt.xlabel('MHz')
        plt.show()

        plt.plot(freqs/units.MHz, att_length)
        plt.show()
#        quit()
        """

#        print(f'ray type {ray_type} and ice position {ant_ice_position[2]} and refl depth {refl_layer_depth}')
        if ray_type == 'reflected' or force_dB or (np.abs(ant_ice_position[2]) > np.abs(refl_layer_depth)):
#            print(f'db adjust triggered')
            efield_adjust *= 10**(-reflective_dB / 20)
#        if force_dB and np.abs(ant_ice_position[2]) > refl_layer_depth:
#            efield_adjust *= 10**(-reflective_dB / 20)

        electric_field.set_frequency_spectrum(electric_field.get_frequency_spectrum() * efield_adjust, electric_field.get_sampling_rate())
#        electric_field.set_parameter(efp.zenith, refr_zenith)

        """
        post_val = 0
        for iF, fre in enumerate(freqs):
            if 190 <= fre/units.MHz <= 200:
                for i in range(20):
                    post_val += np.abs(electric_field.get_frequency_spectrum()[2])[iF+10-i]               
                break 

        if pre_val == 0:
            pre_val = 1
        print(f'pre at {pre_val}, post {post_val}, ratio {post_val/pre_val} compared to 200MHz att of {efield_adjust[iF]} for dist traveled of {dist_traveled} and att length {att_length[iF]}')

        plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[0]), label='Post-att 0')
        plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[1]), label='Post-att 1')
        plt.plot(freqs/units.MHz, np.abs(electric_field.get_frequency_spectrum()[2]), label='Post-att 2')
        plt.plot(freqs/units.MHz,efield_adjust, color='black', label='E field reduction')
        plt.legend()
        plt.xlabel('Freqs (MHz)')
        plt.yscale('log')
        plt.show()
        """

        return electric_field


    @register_run()
#    def run(self, detector, ray_type='direct', antenna_depth=0*units.m, layer_depth=0*units.m, layer_dB=0, output_mode=0):
    def run(self, detector, ray_type='direct', layer_depth=-300*units.m, layer_dB=0, force_dB = False, attenuation_model=None, output_mode=0):
        """
        Read in a random sample of stations from a CoREAS file.
        For each position the closest observer is selected and a simulated
        event is created for that observer.

        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
        ray_type: string
            'direct'    : direct surface observers and surface antennas
            'refracted' : get observer locations based off of refraction to antenna depth
            'reflected' : get observer locations based off of refraction and reflection at a reflective layer
            'by_depth'  : get observer locations depending upon if observer is above or below layer_depth
        antenna_depth: positive float
            Depth of antenna considered for non-direct ray types in meters
        layer_depth: positive float
            Depth of reflective layer for reflected signals in meters
        layer_dB: float
            Relative power loss in reflective layer in dB
        attenuation_model: string (default None)
            Attenuation model to be used.
            'MB_flat' applies a flat attenuation length of 460m - 180m/GHz * frequency
        output_mode: integer (default 0)
            0: only the event object is returned
            1: the function reuturns the event object, the current inputfilename, the distance between the choosen station and the requested core position,
               and the area in which the core positions are randomly distributed
            2: the function returned the event object, the current inputfilename, and the core x and y positions


        """
        while (self.__current_input_file < len(self.__input_files)):
            t = time.time()
            t_per_event = time.time()
            filesize = os.path.getsize(self.__input_files[self.__current_input_file])
            if(filesize < 18456 * 2):  # based on the observation that a file with such a small filesize is corrupt
                self.logger.warning("file {} seems to be corrupt, skipping to next file".format(self.__input_files[self.__current_input_file]))
                self.__current_input_file += 1
                continue
            corsika = h5py.File(self.__input_files[self.__current_input_file], "r")
            self.logger.info(
                "using coreas simulation {} with E={:2g} theta = {:.0f}".format(
                    self.__input_files[self.__current_input_file],
                    corsika['inputs'].attrs["ERANGE"][0] * units.GeV,
                    corsika['inputs'].attrs["THETAP"][0]
                )
            )
            positions = []
            for i, observer in enumerate(corsika['CoREAS']['observers'].values()):
                position = observer.attrs['position']
                positions.append(np.array([-position[1], position[0], 0]) * units.cm)
#                 self.logger.debug("({:.0f}, {:.0f})".format(positions[i][0], positions[i][1]))
            positions = np.array(positions)

            zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)
            positions_vBvvB = cs.transform_from_magnetic_to_geographic(positions.T)
            positions_vBvvB = cs.transform_to_vxB_vxvxB(positions_vBvvB).T
#             for i, pos in enumerate(positions_vBvvB):
#                 self.logger.debug("star shape")
#                 self.logger.debug("({:.0f}, {:.0f}); ({:.0f}, {:.0f})".format(positions[i, 0], positions[i, 1], pos[0], pos[1]))

            dd = (positions_vBvvB[:, 0] ** 2 + positions_vBvvB[:, 1] ** 2) ** 0.5
            ddmax = dd.max()
            self.logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            if self.__shape == 'square':
                # generate core positions randomly within a rectangle
                cores = np.array([self.__random_generator.uniform(self.__area[0], self.__area[1], self.__n_cores),
                                  self.__random_generator.uniform(self.__area[2], self.__area[3], self.__n_cores),
                                  np.zeros(self.__n_cores)]).T
            elif self.__shape == 'triangle':
                #generate core positions randomly within a triangle with vertices (xmin, ymin), (xmax, ymin), (0.5(xmax-xmin), ymax)
                v1 = np.array([self.__area[0], self.__area[2]])
                v2 = np.array([self.__area[1], self.__area[2]])
                v3 = np.array([0.5*(self.__area[1]+self.__area[0]), self.__area[3]])
                #Method using barycentric coordinates to generate a uniform distribution
                x, y = self.__random_generator.random(size=self.__n_cores), self.__random_generator.random(size=self.__n_cores)
                q = np.abs(x - y)
                s, t, u = q, 0.5 * (x + y - q), 1- 0.5 * (q + x + y)
                c_X = s * v1[0] + t * v2[0] + u * v3[0]
                c_Y = s * v1[1] + t * v2[1] + u * v3[1]
                cores = np.stack([c_X, c_Y, np.zeros(self.__n_cores)], axis=1)
            elif self.__shape == 'linear':
                #generate core positions along a line bisecting a rectangle
                x = np.linspace(self.__area[0], self.__area[1], num=self.__n_cores)                
                y = np.linspace(self.__area[2], self.__area[3], num=self.__n_cores)
                cores = np.stack([x, y, np.zeros(self.__n_cores)], axis=1)                                
            elif self.__shape == 'uniform':
                square_cores = int(np.sqrt(self.__n_cores))
                xs = np.linspace(self.__area[0], self.__area[1], num=square_cores)
                ys = np.linspace(self.__area[2], self.__area[3], num=square_cores)
                x = []
                y = []
                for i in range(square_cores):
                    for j in range(square_cores):
                        x.append(xs[i])
                        y.append(ys[j])
#                x = np.array(x)
#                y = np.array(y)
                cores = np.stack([x, y, np.zeros_like(x)], axis=1)
            elif self.__shape == 'radial':
#                diameter = np.sqrt( (self.__area[0] - self.__area[1])**2 + (self.__area[2] - self.__area[3])**2 )
                diameter = self.__area[1] - self.__area[0]
                r = diameter / 2 * np.sqrt(self.__random_generator.random(size=self.__n_cores))
                theta = self.__random_generator.random(size=self.__n_cores) * 2 * np.pi
                x_center = (self.__area[0] + self.__area[1])/2
                y_center = (self.__area[2] + self.__area[3])/2
                x = x_center + r * np.cos(theta)
                y = y_center + r * np.sin(theta)
                cores = np.stack([x, y, np.zeros(self.__n_cores)], axis=1)
            else:
                self.logger.debug(f'Only shapes square and triangle configured, no {self.__shape} setup')
                raise NotImplementedError

            self.__t_per_event += time.time() - t_per_event
            self.__t += time.time() - t

            station_ids = detector.get_station_ids()
            for iCore, core in enumerate(cores):
                ic(core)
                t = time.time()
                evt = NuRadioReco.framework.event.Event(self.__current_input_file, iCore)  # create empty event
                sim_shower = coreas.make_sim_shower(corsika)
                evt.add_sim_shower(sim_shower)
                rd_shower = NuRadioReco.framework.radio_shower.RadioShower(station_ids=station_ids)
                evt.add_shower(rd_shower)
                evt_seen = False

                for station_id in station_ids:
                    # convert into vxvxB frame to calculate closests simulated station to detecor station
                    det_station_position = detector.get_absolute_position(station_id)
                    det_station_position[2] = 0
                    channel_ids = detector.get_channel_ids(station_id)
                    # check if we have any are doing reflected or refracted signals. If not continue as normal
                    station_locations = []
                    if not ray_type == 'direct':
                        layer_power_reduction = 10**(-layer_dB / 20)
                        cr_xmax = corsika['CoREAS'].attrs['DepthOfShowerMaximum']
                        # for each channel, get a station position corresponding to the surface after refraction/reflection
                        for channel_id in channel_ids:
                            channel_dict = detector.get_channel(station_id, channel_id)
                            antenna_depth = np.abs( channel_dict['ant_position_z'])
                            antenna_x = channel_dict['ant_position_x']
                            antenna_y = channel_dict['ant_position_y']
#                            print(f'check : antenna depth {antenna_depth} for channel {channel_id} in station {station_id}')
                            n_ice = 1.78
                            refracted_zenith = np.arcsin( np.sin(zenith) / n_ice) * units.rad
#                            print(f'check :  zenith {zenith} refracted {refracted_zenith}')
                            if ray_type == 'reflected':
                                # Depreciated? Remove possible
                                antenna_depth = 2 * layer_depth - antenna_depth
                            refr_dist_from_station = antenna_depth * np.tan(refracted_zenith)

#                            print(f'check, azimuth is {azimuth}')
                            x_shower_direction = refr_dist_from_station * np.sin(azimuth)
                            y_shower_direction = refr_dist_from_station * np.cos(azimuth)

                            refr_station_x = det_station_position[0] + antenna_x + x_shower_direction
#                            print(f'det x {det_station_position[0]} + ant x {antenna_x} + shower x {x_shower_direction} gets refr x {refr_station_x}')
                            refr_station_y = det_station_position[1] + antenna_y + y_shower_direction
#                            print(f'det y {det_station_position[1]} + ant y {antenna_y} + shower y {y_shower_direction} gets refr y {refr_station_y}')

                            station_locations.append([ [refr_station_x, refr_station_y, 0], [channel_id] ])

                            """	#Bad old code
                            r_core = np.sqrt(cores[iCore][0]**2 + cores[iCore][1]**2)
                            refr_station_x = cores[iCore][0] * refr_dist_from_station / r_core
                            refr_station_y = cores[iCore][1] * refr_dist_from_station / r_core
                            print(f'original core {cores[iCore]}, r_core {r_core}, refr x y {refr_station_x} {refr_station_y}, zen {zenith} refr zen {refracted_zenith}')
                            station_locations.append([[refr_station_x, refr_station_y, 0], [channel_id]])
                            """

                    else:
#                        station_locations.append([ det_station_position, [channel_ids] ] )
                        for channel_id in channel_ids:
                            station_locations.append( [ det_station_position, [channel_id] ] )
                    # station_locations = np.array(station_locations)
                    station_locations = station_locations

#                    print(f'check : core locations {station_locations}')
                    
                    main_station = NuRadioReco.framework.station.Station(station_id)
                    for station_position, ch_ids in station_locations:
#                        core_rel_to_station = core - det_station_position
                        core_rel_to_station = core - station_position
            #             core_rel_to_station_vBvvB = cs.transform_from_magnetic_to_geographic(core_rel_to_station)
                        core_rel_to_station_vBvvB = cs.transform_to_vxB_vxvxB(core_rel_to_station)
                        dcore = (core_rel_to_station_vBvvB[0] ** 2 + core_rel_to_station_vBvvB[1] ** 2) ** 0.5
    #                     print(f"{core_rel_to_station}, {core_rel_to_station_vBvvB} -> {dcore}")

#                        print(f'station check, position {station_position} ch_ids {ch_ids}, core relative to station {core_rel_to_station}')
                        station = NuRadioReco.framework.station.Station(station_id)
                        if(dcore > ddmax):
                            # station is outside of the star shape pattern, create empty station
#                            station = NuRadioReco.framework.station.Station(station_id)
#                            channel_ids = detector.get_channel_ids(station_id)
                            sim_station = coreas.make_sim_station(station_id, corsika, None, ch_ids)
#                            station.set_sim_station(sim_station)

#                            evt.set_station(station)   #OLD

#                            self.logger.debug(f"station {station_id} is outside of star shape, channel_ids {channel_ids}")
                            self.logger.debug(f"station {station_id} is outside of star shape, channel_ids {ch_ids}")
                        else:
                            evt_seen = True
                            distances = np.linalg.norm(core_rel_to_station_vBvvB[:2] - positions_vBvvB[:, :2], axis=1)
                            index = np.argmin(distances)
                            distance = distances[index]
                            key = list(corsika['CoREAS']['observers'].keys())[index]
                            self.logger.debug(
                                f"generating core at ground ({cores[iCore][0]:.0f}, {cores[iCore][1]:.0f}), rel to station{station_id} ({core_rel_to_station[0]}, \n"
                                +f"{core_rel_to_station[1]}) of raw station loc ~({station_locations[0][0]}, {station_locations[0][1]}), \n"
                                +f"vBvvB({core_rel_to_station_vBvvB[0]}, {core_rel_to_station_vBvvB[1]}) with dcore {dcore} and ddmax {ddmax},\n" 
                                +f"nearest simulated station is {distance}m away at ground ({positions[index][0]}, {positions[index][1]}), vBvvB({positions_vBvvB[index][0]}, {positions_vBvvB[index][1]})")
                            t_event_structure = time.time()
                            observer = corsika['CoREAS']['observers'].get(key)

#                            station = NuRadioReco.framework.station.Station(station_id)
#                            channel_ids = detector.get_channel_ids(station_id)
                            sim_station = coreas.make_sim_station(station_id, corsika, observer, ch_ids)
#                            station.set_sim_station(sim_station)

                        if not main_station.has_sim_station():
                            if not ray_type == 'direct':
                                efields = sim_station.get_electric_fields()
                                for efield in efields:
#                                    print(f'efield channels {efield.get_channel_ids()} for chids {ch_ids}')
                                    channel_dict = detector.get_channel(station_id, efield.get_channel_ids()[0])
                                    ant_surf_pos = [channel_dict['ant_position_x'],channel_dict['ant_position_y'],channel_dict['ant_position_z']]
                                    efield = self.modify_eField(efield, station_position, ant_surf_pos, corsika['CoREAS'].attrs['DepthOfShowerMaximum'], 
                                                                ray_type, layer_depth, layer_dB, force_dB, attenuation_model)
                                sim_station.set_electric_fields(efields)
                            main_station.set_sim_station(sim_station)
                            main_sim_station = main_station.get_sim_station()
                        else:
                            for efield in sim_station.get_electric_fields():
#                                print(f'efield params ch ids {efield.get_channel_ids()}')
                                if not ray_type == 'direct':
#                                    print(f'efield channels {efield.get_channel_ids()} for chids {ch_ids}')
                                    channel_dict = detector.get_channel(station_id, efield.get_channel_ids()[0])
                                    ant_surf_pos = [channel_dict['ant_position_x'],channel_dict['ant_position_y'],channel_dict['ant_position_z']]
                                    efield = self.modify_eField(efield, station_position, ant_surf_pos, corsika['CoREAS'].attrs['DepthOfShowerMaximum'], 
                                                                ray_type, layer_depth, layer_dB, force_dB, attenuation_model)
#                                main_station.set_sim_station(sim_station)
                                main_sim_station = main_station.get_sim_station()
                                main_sim_station.add_electric_field(efield)

                    main_station.set_sim_station(main_sim_station)
#                    print(f'check of efields in main station len {len(main_station.get_electric_fields())}')
                    main_station.set_electric_fields(main_sim_station.get_electric_fields())
#                    print(f'check of efields in main station after len {len(main_station.get_electric_fields())}')
#                    print(f'and the sim station efields {len(main_station.get_sim_station().get_electric_fields())}')
                    evt.set_station(main_station)
                if(output_mode == 0):
                    self.__t += time.time() - t
                    yield evt
                elif(output_mode == 1):
                    self.__t += time.time() - t
		#Added a trigger because if the core is thrown where it is not seen on first event, __t_event_structure will not be defined and fail
                    if(evt_seen):
                        self.__t_event_structure += time.time() - t_event_structure
		#This is supposed to also yield distance to chosen station and area cores are distributed in. Given that there are
		#Multiple stations that may be seen with multiple distances, and no saving/indexing of them, unsure goal of this mode
                    yield evt, self.__current_input_file
                elif(output_mode == 2):
                    self.__t += time.time() - t
                    if(evt_seen):
                        self.__t_event_structure += time.time() - t_event_structure
                    yield evt, self.__current_input_file, cores[iCore][0], cores[iCore][1]
                else:
                    self.logger.debug("output mode > 2 not implemented")
                    raise NotImplementedError

            self.__current_input_file += 1

    """
    def get_sim_station_per_observer(station_id, detector, corsika, core, det_station_position):



        core_rel_to_station = core - det_station_position
    #             core_rel_to_station_vBvvB = cs.transform_from_magnetic_to_geographic(core_rel_to_station)
        core_rel_to_station_vBvvB = cs.transform_to_vxB_vxvxB(core_rel_to_station)
        dcore = (core_rel_to_station_vBvvB[0] ** 2 + core_rel_to_station_vBvvB[1] ** 2) ** 0.5
    #                     print(f"{core_rel_to_station}, {core_rel_to_station_vBvvB} -> {dcore}")
        if(dcore > ddmax):
            # station is outside of the star shape pattern, create empty station
            station = NuRadioReco.framework.station.Station(station_id)
            channel_ids = detector.get_channel_ids(station_id)
            sim_station = coreas.make_sim_station(station_id, corsika, None, channel_ids)
            station.set_sim_station(sim_station)
            evt.set_station(station)
            self.logger.debug(f"station {station_id} is outside of star shape, channel_ids {channel_ids}")
        else:
            evt_seen = True
            distances = np.linalg.norm(core_rel_to_station_vBvvB[:2] - positions_vBvvB[:, :2], axis=1)
            index = np.argmin(distances)
            distance = distances[index]
            key = list(corsika['CoREAS']['observers'].keys())[index]
            self.logger.debug(
                "generating core at ground ({:.0f}, {:.0f}), rel to station ({:.0f}, {:.0f}) vBvvB({:.0f}, {:.0f}), nearest simulated station is {:.0f}m away at ground ({:.0f}, {:.0f}), vBvvB({:.0f}, {:.0f})".format(
                    cores[iCore][0],
                    cores[iCore][1],
                    core_rel_to_station[0],
                    core_rel_to_station[1],
                    core_rel_to_station_vBvvB[0],
                    core_rel_to_station_vBvvB[1],
                    distance / units.m,
                    positions[index][0],
                    positions[index][1],
                    positions_vBvvB[index][0],
                    positions_vBvvB[index][1]
                )
            )
            t_event_structure = time.time()
            observer = corsika['CoREAS']['observers'].get(key)

            station = NuRadioReco.framework.station.Station(station_id)
            channel_ids = detector.get_channel_ids(station_id)
            sim_station = coreas.make_sim_station(station_id, corsika, observer, channel_ids)
            station.set_sim_station(sim_station)
            evt.set_station(station)
    """

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt
