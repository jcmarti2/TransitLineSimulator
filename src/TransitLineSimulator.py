import bisect
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import _pickle as pickle
# import pandas as pd

__author__ = 'juan carlos martinez mori'

clk = 0.0
t_list = []
event_list = []


class TransitLineSimulator:

    def __init__(self, max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                 bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                 delays=None, bunch_threshold_s=False, bus_addition_list=None):
        """
        this is the constructor for the simulator class
        :param max_clk_s: maximum simulation time [s]
        :param num_stops: number of stops []
        :param pax_hr: pax per hour at each stop [pax/hr]
        :param stop_spacing_m: spacing between stops [m]
        :param num_buses: number of buses []
        :param headway_s: headway between buses [s]
        :param bus_capacity: bus capacity [pax]
        :param bus_mean_speed_kmh: mean bus cruise speed [km/h]
        :param bus_cv_speed: coefficient of variation for bus speed
        :param bus_mean_acc_ms2: mean bus acceleration [m/s2]
        :param bus_cv_acc: coefficient of variation for bus acceleration
        :param pax_board_s: boarding time per pax [s/pax]
        :param pax_alight_s: alighting time per pax [s/pax]
        :param delays: dictionary of delays (holding data from other replications of same batch)
        :param bunch_threshold_s: threshold for bus addition
        :param bus_addition_list: list of scheduled bus additions
        """

        # save input data
        self.max_clk_s = max_clk_s

        self.num_stops = num_stops
        self.pax_hr = pax_hr
        self.stop_spacing_m = stop_spacing_m

        self.num_buses = num_buses
        self.headway_s = headway_s
        self.bus_capacity = bus_capacity
        self.bus_mean_speed_kmh = bus_mean_speed_kmh
        self.bus_cv_speed = bus_cv_speed
        self.bus_mean_acc_ms2 = bus_mean_acc_ms2
        self.bus_cv_acc = bus_cv_acc
        self.pax_board_s = pax_board_s
        self.pax_alight_s = pax_alight_s

        self.delays = delays
        self.bunch_threshold_s = bunch_threshold_s
        self.bus_addition_list = bus_addition_list

        # store instances
        self.stops = []
        self._build_stops()
        self.buses = []
        self._build_buses()

    def _build_stops(self):
        """
        this function builds the stops in the simulation and saves them in self.stops
        :return:
        """

        for stop_id in range(self.num_stops):

            # get ids of other stops (important for setting pax origin/destination)
            other_stop_ids = list(range(self.num_stops))
            del other_stop_ids[stop_id]

            self.stops.append(Stop(stop_id, self.pax_hr, self.stop_spacing_m, other_stop_ids))

    def _build_buses(self):
        """
        this function builds the buses in the simulation and saves them in self.buses
        :return:
        """

        # build scheduled buses
        for bus_id in range(self.num_buses):
            self.buses.append(Bus(bus_id, self.headway_s, self.num_stops, self.max_clk_s, self.bus_capacity,
                                  self.bus_mean_speed_kmh, self.bus_cv_speed, self.bus_mean_acc_ms2,
                                  self.bus_cv_acc, self.stops, self.pax_board_s, self.pax_alight_s))

    def _initialize(self):
        """
        this function initializes the simulation with prevention of initial transient in pax arrival
        :return:
        """

        global t_list
        global event_list

        # first bus needs to cover all stops
        warmup_timetable = self.buses[0].timetable
        stop_id = 0
        for event in warmup_timetable:
            # pax start arriving at a stop once 'fake' bus arrival happens
            if event[1] == 'arrival' and stop_id < self.num_stops:
                t = event[0] - self.headway_s

                # schedule pax arrival event
                idx = bisect.bisect_left(t_list, t)  # bisect left because pax arrival has priority
                t_list.insert(idx, t)
                event_list.insert(idx, ('pax_arrival', self.stops[stop_id]))

                # go to next stop
                stop_id += 1

            elif stop_id >= self.num_stops:
                break

    def simulate(self):

        global clk
        global t_list
        global event_list

        self._initialize()

        try:
            delays = pickle.load(open(self.pickle_file, "rb"))
        except (OSError, IOError) as e:
            delays = {}
            pickle.dump(delays, open(self.pickle_file, 'wb'))
            delays = pickle.load(open(self.pickle_file, 'rb'))


        # bus_data = {bus.bus_id: {'time': [], 'dist': [], 'delay': [], 'schedule': []}
        #             for bus in self.buses if bus.timetable}

        while t_list and event_list:

            if t_list[0] > self.max_clk_s:
                break

            clk = t_list.pop(0)
            event = event_list.pop(0)

            if event[0] == 'pax_arrival':
                stop = event[1]
                stop.run_pax_arrival()
            elif event[0] == 'bus_arrival':
                bus = event[1]
                bus.run_arrival()
                if bus.timetable_idx < len(bus.timetable):
                    # bus_data[bus.bus_id]['time'].append(clk)
                    # bus_data[bus.bus_id]['dist'].append(bus.travelled_dist)
                    # bus_data[bus.bus_id]['delay'].append(bus.delay)
                    # bus_data[bus.bus_id]['schedule'].append(bus.timetable[bus.timetable_idx - 1][0])
                    schedule = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][0]))
                    dist = float('{0:.2f}'.format(bus.travelled_dist))
                    if (schedule, dist) not in delays:
                        delays[(schedule, dist)] = []
                    delays[(schedule, dist)].append(bus.delay)

            elif event[0] == 'bus_departure':
                bus = event[1]
                bus.run_departure()
                if bus.timetable_idx < len(bus.timetable):
                    # bus_data[bus.bus_id]['time'].append(clk)
                    # bus_data[bus.bus_id]['dist'].append(bus.travelled_dist)
                    # bus_data[bus.bus_id]['delay'].append(bus.delay)
                    # bus_data[bus.bus_id]['schedule'].append(bus.timetable[bus.timetable_idx - 1][0])
                    schedule = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][0]))
                    dist = float('{0:.2f}'.format(bus.travelled_dist))
                    if (schedule, dist) not in delays:
                        delays[(schedule, dist)] = []
                    delays[(schedule, dist)].append(bus.delay)

        pickle.dump(delays, open(self.pickle_file, 'wb'))

        # plt.figure()
        # for bus in bus_data:
        #     if bus_data[bus]:
        #         # plt.plot(bus_data[bus]['delay'])
        #         plt.plot(bus_data[bus]['time'], bus_data[bus]['dist'], color='k')
        #         plt.plot(bus_data[bus]['schedule'], bus_data[bus]['dist'], color='b')
        #
        # plt.ylabel('Distance [m]')
        # plt.xlabel('Time [s]')
        # plt.show()

        # return bus_data


class Bus:

    def __init__(self, bus_id, headway_s, num_stops, max_clk_s, capacity, mean_speed_kmh, cv_speed,
                 mean_acc_ms2, cv_acc, stops, pax_board_s, pax_alight_s,
                 add_bus_s=None, add_bus_dist=None, add_bus_stop_id=None):
        """
        constructor for the buss class
        :param bus_id: bus unique id as an integer []
        :param headway_s: headway between buses [s]
        :param capacity: bus capacity [pax]
        :param mean_speed_kmh: mean bus cruise speed [km/h]
        :param cv_speed: coefficient of variation for bus speed
        :param mean_acc_ms2: mean bus acceleration [m/s2]
        :param cv_acc: coefficient of variation for bus acceleration
        :param pax_board_s: boarding time per pax [s/pax]
        :param pax_alight_s: alighting time per pax [s/pax]
        :param add_bus_s: time when bus is added [s], default is None
        :param add_bus_dist: dist where bus is added [m], default is None
        :param add_bus_stop_id: stop id of where bus is added. default is None
        """

        # save input
        self.bus_id = bus_id
        self.headway_s = headway_s
        self.num_stops = num_stops
        self.max_clk_s = max_clk_s
        self.capacity = capacity
        self.mean_speed_ms = mean_speed_kmh / 3.6
        self.cv_speed = cv_speed
        self.mean_acc_ms2 = mean_acc_ms2
        self.cv_acc = cv_acc
        self.stops = stops
        self.pax_board_s = pax_board_s
        self.pax_alight_s = pax_alight_s

        # lognormal random variables
        std_speed = self.mean_speed_ms * self.cv_speed
        # obtain mean and std of associated normal distribution for the lognormal distributions of speed using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self._mu_speed = np.log((self.mean_speed_ms ** 2) / np.sqrt(std_speed ** 2 + self.mean_speed_ms ** 2))
        self._sigma_speed = np.sqrt(np.log((std_speed ** 2) / (self.mean_speed_ms ** 2) + 1))

        std_acc = self.mean_acc_ms2 * self.cv_acc
        # obtain mean and std of associated normal distribution for the lognormal distributions of acc rate using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self._mu_acc = np.log((self.mean_acc_ms2 ** 2) / np.sqrt(std_acc ** 2 + self.mean_acc_ms2 ** 2))
        self._sigma_acc = np.sqrt(np.log((std_acc ** 2) / (self.mean_acc_ms2 ** 2) + 1))

        # set bus deployment parameters
        if not add_bus_s and not add_bus_dist and not add_bus_stop_id:
            # first bus leaves a headway after simulation starts
            self.deploy_clk_s = (self.bus_id + 1) * self.headway_s
            self.stop_id = -1
            self.travelled_dist = - stops[0].spacing
        else:
            self.deploy_clk_s = add_bus_s
            self.stop_id = (add_bus_stop_id - 1) % self.num_stops
            self.travelled_dist = add_bus_dist - stops[0].spacing

        # build timetable
        self.timetable = self._build_timetable()
        self.timetable_idx = 0

        # build pax storage
        self.pax_lists = {}
        for stop_id in range(self.num_stops):
            self.pax_lists[stop_id] = []
        self.num_pax = 0

        # variables for service state
        self.delay = 0
        self.in_service = False

        # schedule first bus arrival
        self.schedule_arrival(first_stop=True)

    def _build_timetable(self):
        """
        this function builds the bus timetable (used for delay measurement)
        :return:
        """

        timetable = []

        # start at first scheduled stop
        t = self.deploy_clk_s
        stop_id = (self.stop_id + 1) % self.num_stops

        while True:

            if t < self.max_clk_s:

                # time of arrival
                timetable.append((t, 'arrival', stop_id))

                # compute expected time for next departure
                t += max((self.stops[stop_id].pax_s * self.pax_alight_s * self.headway_s),
                         (self.stops[stop_id].pax_s * self.pax_board_s * self.headway_s))

            if t < self.max_clk_s:
                # time of departure
                timetable.append((t, 'departure', stop_id))

                # compute expected time for next arrival
                v = self.mean_speed_ms
                a = self.mean_acc_ms2
                s = self.stops[stop_id].spacing

                t += self._time_inter_stop(v, a, s)
                stop_id = (stop_id + 1) % self.num_stops

            else:
                break

        return tuple(timetable)

    def schedule_departure(self, num_pax_boarding, num_pax_alighting):
        """
        this function schedules a bus departure
        :param num_pax_boarding: number of pax boarding bus []
        :param num_pax_alighting: number of pax alighting bus []
        :return:
        """

        global clk
        global t_list
        global event_list

        if self.in_service and self.timetable_idx < len(self.timetable):
            t_board = num_pax_boarding * self.pax_board_s  # time for all pax to alight
            t_alight = num_pax_alighting * self.pax_alight_s  # time for all pax to board

            # note that departure timetable is with respect to ideal case, so no + clk is needed
            t = max(t_board, t_alight, self.timetable[self.timetable_idx][0] - clk)

            idx = bisect.bisect_left(t_list, clk + t)
            t_list.insert(idx, clk + t)
            event_list.insert(idx, ('bus_departure', self))

    def schedule_arrival(self, first_stop=False):
        """
        this function schedules bus stop arrival
        :param first_stop: True if first stop of bus, False is default
        :return:
        """

        global clk
        global t_list
        global event_list

        if (self.in_service or first_stop) and self.timetable_idx < len(self.timetable):
            if not first_stop:
                # random
                v = np.random.lognormal(self._mu_speed, self._sigma_speed)
                a = np.random.lognormal(self._mu_acc, self._sigma_acc)
                s = self.stops[self.stop_id].spacing
                dt = self._time_inter_stop(v, a, s)
            else:
                # on-time based on timetable
                dt = self.timetable[self.timetable_idx][0] - clk

            idx = bisect.bisect_left(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_arrival', self))

    def run_departure(self):

        global clk

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

        # schedule next event
        self.timetable_idx += 1
        if self.timetable_idx < len(self.timetable):
            self.schedule_arrival()

    def run_arrival(self):
        """
        this function run a bus arrival to a stop
        :return:
        """

        global clk

        self.stop_id = (self.stop_id + 1) % self.num_stops
        self.travelled_dist += self.stops[self.stop_id].spacing

        # put bus in service if needed
        if self.timetable_idx == 0:
            self.in_service = True

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

        # run pax processing
        num_pax_boarding, num_pax_alighting = self._process_pax()

        # shedule next event
        self.timetable_idx += 1
        if self.timetable_idx < len(self.timetable):
            self.schedule_departure(num_pax_boarding, num_pax_alighting)

    def _process_pax(self):
        """
        this function process pax boarding/alighting at a stop
        :return num_pax_boarding: number of pax boarding the bus
        :return num_pax_alighting: number of pax alighting the bus
        """

        num_pax_alighting = len(self.pax_lists[self.stop_id])
        self.num_pax -= num_pax_alighting
        self.pax_lists[self.stop_id] = []

        on_pax = self.stops[self.stop_id].pax
        num_pax_boarding = 0
        for pax in on_pax:
            if self.num_pax < self.capacity:
                num_pax_boarding += 1
                self.num_pax += 1
                self.pax_lists[pax.destination].append(pax)
            else:
                break

        # remove the boarding passengers from the stop
        self.stops[self.stop_id].board_pax(num_pax_boarding)

        return num_pax_boarding, num_pax_alighting

    @staticmethod
    def _time_inter_stop(v, a, s):
        """
        this function computes the time to travel between stops given instantiated variables
        :param v: velocity [m/s]
        :param a: acceleration [m/s2]
        :param s: spacing [m]
        :return t: time to travel [s]
        """

        # if spacing allows for cruise speed
        if s >= (v ** 2) / (2 * a):
            acc_s = (v ** 2) / (2 * a)  # [m]
            cruise_s = s - (v ** 2) / (2 * a)  # [m]
            t = cruise_s / v + (2 / (2 ** 0.5)) * ((acc_s / a) ** 0.5)  # [s]
        # if spacing does not allow for cruise speed
        else:
            t = (2 / (2 ** 0.5)) * ((s / a) ** 0.5)  # [s]

        return t


class Stop:

    def __init__(self, stop_id, pax_hr, spacing, other_stop_ids):

        self.stop_id = stop_id
        self.pax_s = pax_hr / 3600
        self.spacing = spacing
        self.other_stop_ids = tuple(other_stop_ids)

        self.pax = []

    def schedule_pax_arrival(self):

        global clk
        global t_list
        global event_list

        t = np.random.exponential(1/self.pax_s)
        idx = bisect.bisect_left(t_list, clk + t)
        t_list.insert(idx, clk + t)
        event_list.insert(idx, ('pax_arrival', self))

    def run_pax_arrival(self):
        destination = np.random.choice(self.other_stop_ids)
        self.pax.append(Pax(self.stop_id, destination))
        self.schedule_pax_arrival()  # based on current clock time

    def board_pax(self, num_pax_boarding):
        del self.pax[0:num_pax_boarding]


# namedtuple for passengers
Pax = namedtuple('Pax', ['origin', 'destination'])
