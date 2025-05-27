class DORISBeacon: 
    def __init__(self, ID:int, station_code:str, name:str, domes:str, type:str, freq_shift:int) -> None:
        self.ID = ID
        self.station_code = station_code
        self.name = name
        self.domes = domes
        self.type = type
        self.freq_shift = freq_shift # station frequency shift factor

        self.time_ref_bit = False
        self.time_bias = 0.0 # time beacon reference vs. TAI reference time [1e-6 s]
        self.time_shift = 0.0 # time beacon reference shift [1e-14 s/s]

class Station:
    def __init__(self, station_code: str):
        self.station_code = station_code
        self.soln_epochlist = []  # list[Timestamp]: Different epochs for the station
        self.soln_coor = []  # list[list[float]]: XYZ coordinates for each epoch
        self.antenna_types = []  # list[str]: Antenna type for each epoch
        self.soln_idx = []  # list[int]: Solution indices matching epochs


