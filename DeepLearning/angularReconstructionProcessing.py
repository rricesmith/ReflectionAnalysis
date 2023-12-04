import NuRadioReco.modules.correlationDirectionFitter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.detector import generic_detector
import argparse










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('filename', type=str, help='File to run reconstruction on')
    parser.add_argument('station_id', type=int, help='Id of station being reconstruted')
    args = parser.parse_args()
    filename = args.filename
    station_id = args.station_id

    det = generic_detector.GenericDetector(json_filename=f'DeepLearning/station_configs/station{station_id}.json', assume_inf=False, antenna_by_depth=False, default_station=station_id)
    reader = NuRadioRecoio.NuRadioRecoio(filename)

    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    correlationDirectionFitter.begin(debug=False)


    for i, evt in enumerate(reader.get_events()):
        station = evt.get_station(station_id)
        correlationDirectionFitter.run(evt, station, det, n_index=1.35)


    reader.close_files()

