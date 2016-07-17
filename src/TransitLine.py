import os
import time
import bisect
import numpy as np

__author__ = 'Juan Carlos Martinez Mori'


clk = 0
t_list = []
event_list = []


class TransitLine:

    def __init__(self):
        """
        this is the constructor for the transit line class
        relevant directories are set
        everything else starts as None
        """

        print('Status: Initializing TransitLine ...')

        # save directories in class
        self._config_dir = '{0}/../config/'.format(os.path.dirname(os.path.realpath(__file__)))
        self._output_dir = '{0}/../output/'.format(os.path.dirname(os.path.realpath(__file__)))

        # configuration files and output files are not loaded yet
        self._stops_config_file = None
        self._buses_config_file = None
        self._output_file = None

        # stops and buses list
        self._max_clk = None
        self._stops = None
        self._buses = None

    def simulate(self, rep_id, max_clk, save_bus=False, save_stop=False):
        """
        this function runs the simulation
        :param rep_id: replication id
        :param max_clk: max clock time of this simulation
        :param save_bus: boolean to write bus simulation output in output file
        :param save_stop: boolean to write stop simulation output in output file
        :return:
        """

        # set configuration and result files
        self._stops_config_file = '{0}rep{1}_stops.txt'.format(self._config_dir, rep_id)
        self._buses_config_file = '{0}rep{1}_buses.txt'.format(self._config_dir, rep_id)
        self._output_file = '{0}rep{1}_output.txt'.format(self._output_dir, rep_id)

        # set max clock time
        self._max_clk = max_clk

        # parse configuration files
        self._parse_config_files()

        # run simulation
        self._simulate(rep_id, save_bus, save_stop)

        # reset class
        self._reset_class()

    def _simulate(self, rep_id, save_bus, save_stop):

        print('Status: Simulating rep{0} ...'.format(rep_id))
        time0 = time.time()

        global clk
        global t_list
        global event_list

        # build output file if it does not yet exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        with open(self._output_file, 'w+') as output_file:
            ent = 0
            while t_list and event_list:

                # if the next event is beyond max clock time, end simulation
                if t_list[0] > self._max_clk:
                    break

                # update clock time
                clk = t_list[0]

                if isinstance(event_list[0], Stop) and save_stop:
                    event_type = 3
                    stop_id, stop_pax = event_list[0].run_pax_arrival()
                    bus_id = ''
                    bus_type = ''
                    bus_pax = ''
                elif isinstance(event_list[0], Bus) and save_bus:
                    pass

                line = '{0},{1:.2f},{2},{3},{4},{5},{6},{7}\n'.format(ent, clk, event_type, stop_id,
                                                                  stop_pax, bus_id, bus_type, bus_pax)
                output_file.write(line)
                ent += 1

                t_list.pop(0)
                event_list.pop(0)

        time1 = time.time()
        print('Status: Finished simulating rep{0} ...'.format(rep_id))
        print('        Elapsed time -- {0:.2f} s'.format(time1 - time0))

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
            print(self._stops_config_file)
            raise Exception('At least one configuration file is missing ...')

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
        subseq_alight_demand = []
        for i in stop_data[3].strip('\n').strip('[]').split(','):
            if i:
                subseq_alight_demand.append(float(i)/3600)
        sum_subseq = sum(subseq_alight_demand)
        num_subseq = len(subseq_alight_demand)

        # this is the probability of a passenger going to each of the subsequent stops
        self._subseq_alight_probs = [i/sum_subseq for i in subseq_alight_demand]
        self._subseq_stop_ids = [self._stop_id + i for i in range(1, num_subseq + 1)]

        # the stop starts with no pax
        self._pax = []  # will hold instances of pax class

        # FOR DEBUG
        self.schedule_pax_arrival()

    def schedule_pax_arrival(self):
        """
        this function schedules the next passenger arrival by inserting in t_list and event_list
        :return:
        """

        if self._board_demand:
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

        return self._stop_id, len(self._pax)


class Pax:

    def __init__(self, origin, destination):
        """
        this is the constructor of the passenger class
        :param origin: origin stop id
        :param destination: destination stop id
        """
        self._origin = origin
        self._destination = destination

