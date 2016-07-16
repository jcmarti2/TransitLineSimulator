import os
import bisect
import numpy as np

__author__ = 'Juan Carlos Martinez Mori'


clk = 0
t_list = []
event_list = []


class TransitLine:

    def __init__(self, config_dir, output_dir):
        """
        this is the constructor for the transit line class
        :param config_dir: directory of configuration files
        :param output_dir: directory of output files
        """

        # save directories in class
        self._config_dir = config_dir
        self._output_dir = output_dir

        # configuration files and output files are not loaded yet
        self._stops_config_file = None
        self._buses_config_file = None
        self._output_file = None

        # stops and buses list
        self._stops = None
        self._buses = None

    def simulate(self, rep_id):
        """
        this function runs the simulation
        :param rep_id: replication id
        :return:
        """

        # set configuration and result files
        self._stops_config_file = '{0}rep{1}_stops.txt'.format(self._config_dir, rep_id)
        self._buses_config_file = '{0}rep{1}_buses.txt'.format(self._config_dir, rep_id)
        self._output_file = '{0}rep{1}_output.txt'.format(self._output_dir, rep_id)

        # run simulation
        self._simulate()

        # reset class
        self._reset_class()

    def _simulate(self):

        global clk
        global t_list
        global event_list

        # build output file if it does not yet exist
        if not os.path.exists(self._output_file):
            os.makedirs(self._output_file)

        with open(self._output_file, 'w') as output_file:
            ent = 0
            while t_list and event_list:

                # update clock time
                clk = t_list[0]

                if isinstance(event_list[0], Stop):
                    event_list[0].run_pax_arrival()
                elif isinstance(event_list[0], Bus):
                    pass

                output_file.write(ent)
                ent += 1

                t_list.pop(0)
                event_list.pop(0)

    def _reset_class(self):
        """
        this function resets the class attributes
        :return:
        """
        print('Status: Resetting class ...')
        # reset class attributes
        self._stops_config_file = None
        self._buses_config_file = None
        self._result_file = None
        self._timetable = None
        self._stops = None

    def _parse_config_files(self):
        """
        this function parses the configuration files
        :return:
        """

        print('Status: Parsing configuration files ...')

        if not os.path.exists(self._stops_config_file) or not os.path.exists(self._buses_config_file):
            raise Exception('Configuration file is missing ...')

        # build stops list
        print('        Building stops ...')
        self._stops = []
        with open(self._stops_config_file, 'r') as stops_config_file:
            for stop_data in stops_config_file:
                stop_data = [i for i in stop_data.split(';')]
                # construct a Stop instance and append to stops list
                self._stops.append(Stop(stop_data))

        # build buses list
        print('        Building buses ...')
        self._stops = []
        with open(self._buses_config_file, 'r') as buses_config_file:
            for bus_data in buses_config_file:
                bus_data = [i for i in bus_data.split(';')]
                # construct a Bus instance and append to buses list
                self._buses.append(Bus(bus_data))


class Bus:

    def __init__(self, bus_data):
        """
        this is the constructor for the bus class
        :param bus_data: list holding bus data [bus_id, bus_capacity, cruise_speed, acc_rate]
        """

        self._bus_id = int(bus_data[0])  # int
        self._bus_capacity = int(bus_data[1])  # int
        self._cruise_speed = float(bus_data[2])/3.6  # converted from [km/h] to [m/s]
        self._acc_rate = float(bus_data[3])  # [m/s^2]

        # the bus starts with zero passengers
        self._num_pax = 0  # int

        # timetables
        self.__arr_timetable = []
        self._dept_timetable = []


class Stop:

    def __init__(self, stop_data):
        """
        this is the constructor for the stop class
        :param stop_data: list holding stop data [stop_id, abs_distance, board_demand, [subseq_alight_demand]]
        """

        self._stop_id = int(stop_data[0])  # int
        self._abs_dist = float(stop_data[1])  # [m]
        # this is the boarding demand of this station
        self._board_demand = float(stop_data[2])/3600  # converted from [pax/hr] to [pax/s]

        # this is the alight demand of the subsequent stations coming from the current stations
        # note that self._board_demand MUST be equal to sum(self._subseq_alight_demand)
        # converted from [pax/hr] to [pax/s]
        subseq_alight_demand = [float(i)/3600 for i in stop_data[3].strip("[]").split(',')]
        sum_subseq = sum(subseq_alight_demand)
        num_subseq = len(subseq_alight_demand)
        # this is the probability of a passenger going to each of the subsequent stops
        self._subseq_alight_probs = [i/sum_subseq for i in subseq_alight_demand]
        self._subseq_stop_ids = [self._stop_id + i for i in range(1, num_subseq + 1)]

        # the stop starts with no pax
        self._pax = []  # will hold instances of pax class

    def schedule_pax_arrival(self):
        """
        this function schedules the next passenger arrival by inserting in t_list and event_list
        :return:
        """
        t = np.random.exponential(1/self._board_demand)
        idx = bisect.bisect_left(t_list, clk + t)
        t_list.insert(idx, clk + t)
        event_list.insert(idx, self)

    def run_pax_arrival(self):
        """
        this function inserts one pax and schedules next pax arrival
        :return:
        """
        # compute destination station
        destination = np.random.choice(self._subseq_stop_ids, p=self._subseq_alight_probs)

        # append to pax list
        self._pax.append(Pax(self._stop_id, destination))

        # schedule next passenger arrival
        self.schedule_pax_arrival()


class Pax:

    def __init__(self, origin, destination):
        """
        this is the constructor of the passenger class
        :param origin: origin stop id
        :param destination: destination stop id
        """
        self._origin = origin
        self._destination = destination

