import os
import time
import bisect
import numpy as np

__author__ = 'Juan Carlos Martinez Mori'


clk = 0
t_list = []
event_list = []
stops = {}
buses = {}


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
        self._reset_simulator()

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
            ent = 0  # output entry counter

            # initialize simulation
            self._initialize()
            event_type = 1  # initialize is event_type 1
            stop_id, stop_pax = '', ''
            bus_id, bus_type, bus_pax = '', '', ''
            line = '{0},{1:.2f},{2},{3},{4},{5},{6},{7}\n'.format(ent, clk, event_type, stop_id,
                                                                  stop_pax, bus_id, bus_type, bus_pax)
            output_file.write(line)  # writing initialize event
            ent += 1

            print('        Running ...')
            while t_list and event_list:

                # if the next event is beyond max clock time, end simulation
                if t_list[0] > self._max_clk:
                    break

                # update clock time
                clk = t_list[0]

                # event type is passenger arrival
                if isinstance(event_list[0], Stop):
                    event_type = 3  # pax arrival event type
                    event_list[0].run_pax_arrival()
                    stop_id = event_list[0].stop_id
                    stop_pax = event_list[0].num_pax
                    if save_stop:
                        bus_id, bus_type, bus_pax = '', '', ''
                        line = '{0},{1:.2f},{2},{3},{4},{5},{6},{7}\n'.format(ent, clk, event_type, stop_id,
                                                                              stop_pax, bus_id, bus_type, bus_pax)
                        output_file.write(line)

                elif isinstance(event_list[0][0], Bus):

                    # event type is bus arrival to stop
                    if event_list[0][1] == 'arrival':
                        event_type = 1  # bus arrival event type
                        stop_id, stop_pax, bus_id, bus_type, bus_pax = event_list[0][0].run_arrival()
                        if save_bus:
                            line = '{0},{1:.2f},{2},{3},{4},{5},{6},{7}\n'.format(ent, clk, event_type, stop_id,
                                                                                  stop_pax, bus_id, bus_type, bus_pax)
                            output_file.write(line)

                    # event type is bus departure from stop
                    elif event_list[0][1] == 'departure':
                        event_type = 2  # bus departure event type
                        stop_id, stop_pax, bus_id, bus_type, bus_pax = event_list[0][0].run_departure()
                        if save_bus:
                            line = '{0},{1:.2f},{2},{3},{4},{5},{6},{7}\n'.format(ent, clk, event_type, stop_id,
                                                                                  stop_pax, bus_id, bus_type, bus_pax)
                            output_file.write(line)

                # remove first items from t_list and event_list
                t_list.pop(0)
                event_list.pop(0)

                ent += 1  # update entry counter

        time1 = time.time()
        print('Status: Finished simulating rep{0} ...'.format(rep_id))
        print('        Elapsed time -- {0:.2f} s'.format(time1 - time0))

    def _reset_simulator(self):
        """
        this function resets the class attributes
        :return:
        """

        print('Status: Resetting simulator ...')

        global clk
        global t_list
        global event_list
        global stops
        global buses

        # reset class attributes
        self._stops_config_file = None
        self._buses_config_file = None
        self._result_file = None
        self._timetable = None

        clk = 0
        t_list = []
        event_list = []
        stops = {}
        buses = {}

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
        with open(self._stops_config_file, 'r') as stops_config_file:
            for stop_data in stops_config_file:
                stop_data = [i for i in stop_data.split(';')]
                # construct a Stop instance and append to stops list
                stops[stop_data[0]] = Stop(stop_data)

        # build buses list
        print('        Building buses ...')
        with open(self._buses_config_file, 'r') as buses_config_file:
            for bus_data in buses_config_file:
                bus_data = [i for i in bus_data.split(';')]
                # construct a Bus instance and append to buses list
                buses[bus_data[0]] = Bus(bus_data)

    def _initialize(self):
        """
        this function initializes the simulation events
        :return:
        """

        global clk
        global t_list
        global event_list

        print('        Initializing ...')

        for bus in self._buses:
            bus.schedule_departure()


class Bus:

    def __init__(self, bus_data):
        """
        this is the constructor for the bus class
        :param bus_data: list holding bus data [bus_id, bus_capacity, mean_cruise_speed, cv_cruise_speed, acc_rate
                                                cv_acc_rate, stop_list, stop_slack, departure_time]
        """

        self._bus_id = int(bus_data[0])  # int
        self._bus_capacity = int(bus_data[1])  # int

        # this section uses bus_data[2] and bus_data[3]
        # cruise speed random distribution
        mean_cruise_speed = float(bus_data[2])/3.6  # converted from [km/h] to [m/s]
        std_cruise_speed = mean_cruise_speed*float(bus_data[3])
        # obtain mean and std of associated normal distribution for the lognormal distributions of speed using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        # [m/s]
        self._mu_cruise_speed = np.log(
            (mean_cruise_speed ** 2) / np.sqrt(std_cruise_speed ** 2 + mean_cruise_speed ** 2))
        self._sigma_cruise_speed = np.sqrt(np.log((std_cruise_speed ** 2) / (mean_cruise_speed ** 2) + 1))

        # this section uses bus_data[4] and bus_data[6]
        # acceleration rate random distribution
        mean_acc_rate = float(bus_data[4])  # [m/s^2]
        std_acc_rate = mean_acc_rate*float(bus_data[5])
        # obtain mean and std of associated normal distribution for the lognormal distributions of acc rate using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        # [m/s^2]
        self._mu_acc_rate = np.log((mean_acc_rate ** 2) / np.sqrt(std_acc_rate ** 2 + mean_acc_rate ** 2))
        self._sigma_acc_rate = np.sqrt(np.log((std_acc_rate ** 2) / (mean_acc_rate ** 2) + 1))

        # this section uses bus_data[6]
        # generate stop list for this bus
        self._stop_list = []
        for i in bus_data[6].strip('\n').strip('[]').split(','):
            if i:
                self._stop_list.append(int(i))
        if not self._stop_list:
            raise Exception('A bus was generated without a stop list ...')

        # this section uses bus_data[7]
        # generate stop list for this bus
        self._stop_slack = []
        for i in bus_data[7].strip('\n').strip('[]').split(','):
            if i:
                self._stop_slack.append(float(i))
        if not self._stop_list:
            raise Exception('A bus was generated without a defined stop slack ...')

        # save departure time
        self._start_time = bus_data[8]

        # attribute to keep track of current stop index
        self._stop_idx = -1

        # the bus starts with zero passengers
        self._pax_lists = {}
        self._num_pax = 0

        # timetables
        self._arr_timetable = []
        self._dept_timetable = []

    def schedule_arrival(self):
        """
        this function schedules a bus arrival to a stop
        if the bus has just been put in service, arrival time to stop is start time
        else, compute random variable based on distance to next stop
        :return:
        """

        global clk
        global t_list
        global event_list

        # is bus was just put in service
        if self._stop_idx < 0:
            t = self._start_time
        else:
            v = np.random.lognormal(self._mu_cruise_speed, self._sigma_cruise_speed)  # [m/s]
            a = np.random.lognormal(self._mu_acc_rate, self._sigma_acc_rate)  # [m/s^2]
            s = self._stop_list[self._stop_idx + 1].abs_dist - self._stop_list[self._stop_idx].abs_dist  # [m]
            # if spacing allows for cruise speed
            if s >= (v**2) / (2*a):
                acc_s = (v**2) / (2*a)  # [m]
                cruise_s = s - (v**2) / (2*a)  # [m]
                t = cruise_s/v + (2/(2**0.5))*((acc_s/a)**0.5)  # [s]
            # if spacing does not allow for cruise speed
            else:
                t = (2/(2**0.5))*((s/a)**0.5)  # [s]

        idx = bisect.bisect_left(t_list, clk + t)
        t_list.insert(idx, clk + t)
        event_list.insert(idx, [self, 'arrival'])

    def schedule_departure(self, num_pax_off, num_pax_on):

        # TODO: set time according to pax alighting and boarding and timetable

        global clk
        global t_list
        global event_list

        t = num_pax_off

        idx = bisect.bisect_left(t_list, clk)
        t_list.insert(idx, clk)
        event_list.insert(idx, [self, 'departure'])

    def _build_timetable(self):
        self._arr_timetable = []
        self._dept_timetable = []

    def run_arrival(self):

        global stops

        # update stop idx
        self._stop_idx += 1

        # remove alighting passengers from bus
        num_pax_off = len(self._pax_lists[self._stop_idx])
        self._num_pax -= num_pax_off
        self._pax_lists[self._stop_idx] = []

        # find passengers that want to board bus at this stop
        on_pax = stops[self._stop_idx].pax
        num_pax_on = 0
        for pax in on_pax:
            if self._num_pax < self._bus_capacity:
                # let new passenger board
                num_pax_on += 1
                self._num_pax += 1
                self._pax_lists[pax.destination].append(pax)
            else:
                break
        # remove the boarding passengers from the stop
        stops[self._stop_idx].run_pax_boarding(num_pax_on)

        # schedule bus departure from stop based on time it will take for pax to board and alight
        # TODO: TAKE CARE OF PAX THAT ARRIVE WHILE PAX ARE BOARDING
        self.schedule_departure(num_pax_off, num_pax_on)

    def run_departure(self):

        self.schedule_arrival()


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
                subseq_alight_demand.append(float(i)/3600)  # applying conversion factor
        sum_subseq = sum(subseq_alight_demand)
        num_subseq = len(subseq_alight_demand)

        # this is the probability of a passenger going to each of the subsequent stops
        # if subset_alight_demand is [], these are also []
        self._subseq_alight_probs = [i/sum_subseq for i in subseq_alight_demand]
        self._subseq_stop_ids = [self._stop_id + i for i in range(1, num_subseq + 1)]

        # the stop starts with no pax
        self._pax = []  # will hold instances of pax class
        self._num_pax = 0

        # FOR DEBUG
        self.schedule_pax_arrival()

    @property
    def stop_id(self):
        return self._stop_id

    @property
    def abs_dist(self):
        return self._abs_dist

    @property
    def pax(self):
        return self._pax

    @property
    def num_pax(self):
        return self._num_pax

    def schedule_pax_arrival(self):
        """
        this function schedules the next passenger arrival by inserting in t_list and event_list
        :return:
        """

        global clk
        global t_list
        global event_list

        # schedule pax arrival only if there is demand
        if self._board_demand:
            t = np.random.exponential(1/self._board_demand)
            idx = bisect.bisect_left(t_list, clk + t)
            t_list.insert(idx, clk + t)
            event_list.insert(idx, self)

    def run_pax_arrival(self):
        """
        this function inserts one pax and schedules next pax arrival
        :return self._stop_id: stop id
        :return len(self._pax): number of pax in the stop
        """
        # compute destination station
        destination = np.random.choice(self._subseq_stop_ids, p=self._subseq_alight_probs)

        # append to pax list
        self._pax.append(Pax(self._stop_id, destination))
        self._num_pax += 1

        # schedule next passenger arrival
        self.schedule_pax_arrival()

    def run_pax_boarding(self, num_pax_boarding):
        """
        this function removes a number of pax from the stop as they board the bus
        :param num_pax_boarding: number of pax boarding a bus
        :return:
        """

        # remove first num_pax_boarding pax from the pax list
        del self._pax[0:num_pax_boarding]
        self._num_pax -= num_pax_boarding



class Pax:

    def __init__(self, origin, destination):
        """
        this is the constructor of the passenger class
        :param origin: origin stop id
        :param destination: destination stop id
        """
        self._origin = origin
        self._destination = destination

    @property
    def origin(self):
        return self._origin

    @property
    def destination(self):
        return self._destination

