









nur_files = []

reader = NuRadioRecoio.NuRadioRecoio(nur_files)


#Key 1 is zen, second is azi, third is energy of CR
template = {}


for i, evt in enumerate(reader.get_events()):
    sim_shower = evt.get_sim_shower(0)
    sim_energy = sim_shower[shp.energy]
    sim_zen = sim_shower[shp.zenith]
    sim_azi = sim_shower[shp.azimuth]

    sim_energy = round(np.log10(sim_energy), 1)
    

    max_trace = [0]
    for ChId, channel in enumerate(station.iter_channels(use_channels=[4, 5, 6, 7])):
        trace = channel.get_trace()
        if max(trace) > max(max_trace):
            max_trace = trace