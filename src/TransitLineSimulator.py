import bisect
import matplotlib.pyplot as plt
import copy
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
                 allow_early, slack_s, delays=None, bunch_threshold_s=False, bus_addition_list=None):
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

        self.allow_early = allow_early
        self.slack_s = slack_s

        self.delays = delays if delays else {}
        self.bunch_threshold_s = bunch_threshold_s
        self.bus_addition_list = bus_addition_list

        if bus_addition_list:
            self.delay_start_s = bus_addition_list[-1][0]
        else:
            self.delay_start_s = 0

        # store instances
        self.stops = {}
        self._build_stops()
        self.buses = {}
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

            self.stops[stop_id] = (Stop(stop_id, self.pax_hr, self.stop_spacing_m, other_stop_ids))

    def _build_buses(self):
        """
        this function builds the buses in the simulation and saves them in self.buses
        :return:
        """

        # build scheduled buses
        for bus_id in range(self.num_buses):
            self.buses[bus_id] = (Bus(bus_id, self.headway_s, self.num_stops, self.max_clk_s, self.bus_capacity,
                                      self.bus_mean_speed_kmh, self.bus_cv_speed, self.bus_mean_acc_ms2,
                                      self.bus_cv_acc, self.stops, self.pax_board_s, self.pax_alight_s,
                                      self.allow_early, self.slack_s))

    def _initialize(self):
        """
        this function initializes the simulation with prevention of initial transient in pax arrival
        it also schedules bus additions
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

        self._schedule_additions()

    def _schedule_additions(self):
        """
        this function schedules the bus additions
        :return:
        """
        global t_list
        global event_list

        if self.bus_addition_list:
            for idx, bus_addition in enumerate(self.bus_addition_list):
                time = bus_addition[0]
                retiring_id = bus_addition[2]
                self.buses[retiring_id].schedule_retirement(time)

    def simulate(self):

        global clk
        global t_list
        global event_list

        self._initialize()

        trajs = {}
        sched = {}
        timetables = {}
        for bus_id in self.buses:
            trajs[bus_id] = []
            sched[bus_id] = []
            timetables[bus_id] = copy.deepcopy(self.buses[bus_id].timetable)

        while t_list and event_list:

            if t_list[0] > self.max_clk_s:
                break

            clk = t_list.pop(0)
            event = event_list.pop(0)

            if event[0] == 'pax_arrival':
                stop = event[1]
                stop.run_pax_arrival()
            elif event[0] == 'pax_flush':
                stop = event[1]
                stop.run_pax_flush()

            elif event[0] == 'bus_arrival':
                bus = event[1]

                # if bus.timetable != timetables[bus.bus_id]:
                #     input('issue')

                if bus.timetable_idx < len(bus.timetable) and bus.in_service:

                    if bus.bus_id == 1:
                        print('arrival at {0}'.format(clk))

                    bus.run_arrival()

                    schedule = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][0]))
                    dist_s = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][3]))
                    dist = float('{0:.2f}'.format(bus.travelled_dist))

                    trajs[bus.bus_id].append((clk, dist))
                    sched[bus.bus_id].append((schedule, dist_s))

                    if schedule > self.delay_start_s:
                        if (schedule, dist, bus.bus_id, bus.timetable_idx) not in self.delays:
                            self.delays[(schedule, dist, bus.bus_id, bus.timetable_idx)] = []
                        self.delays[(schedule, dist, bus.bus_id, bus.timetable_idx)].append(bus.delay)

                    if self.bunch_threshold_s:

                        if bus.delay >= self.bunch_threshold_s:
                            trajs[bus.bus_id].append((None, None))
                            remove_idx = None
                            for idx, item in enumerate(event_list):
                                if item[0] == 'bus_arrival' or item[0] == 'bus_departure':
                                    if item[1].bus_id == bus.bus_id:
                                        remove_idx = idx
                                        break
                            if remove_idx:
                                del t_list[remove_idx]
                                del event_list[remove_idx]

                            timetable_idx = bus.timetable_idx
                            while bus.timetable[timetable_idx][0] < clk:
                                timetable_idx += 1

                            self.buses[bus.bus_id].run_retirement(timetable_idx + 1)

            elif event[0] == 'bus_departure':

                bus = event[1]
                if bus.timetable_idx < len(bus.timetable) and bus.in_service:

                    if bus.bus_id == 1:
                        print('departure at {0}'.format(clk))

                    bus.run_departure()

                    schedule = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][0]))
                    dist_s = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx - 1][3]))
                    dist = float('{0:.2f}'.format(bus.travelled_dist))

                    trajs[bus.bus_id].append((clk, dist))
                    sched[bus.bus_id].append((schedule, dist_s))

                    if schedule > self.delay_start_s:
                        if (schedule, dist, bus.bus_id, bus.timetable_idx) not in self.delays:
                            self.delays[(schedule, dist, bus.bus_id, bus.timetable_idx)] = []
                        self.delays[(schedule, dist, bus.bus_id, bus.timetable_idx)].append(bus.delay)

                    if self.bunch_threshold_s:
                        if bus.delay >= self.bunch_threshold_s:
                            trajs[bus.bus_id].append((None, None))
                            remove_idx = None
                            for idx, item in enumerate(event_list):
                                if item[0] == 'bus_arrival' or item[0] == 'bus_departure':
                                    if item[1].bus_id == bus.bus_id:
                                        remove_idx = idx
                                        break
                            if remove_idx:
                                del t_list[remove_idx]
                                del event_list[remove_idx]

                            timetable_idx = bus.timetable_idx
                            while bus.timetable[timetable_idx][0] < clk:
                                timetable_idx += 1

                            self.buses[bus.bus_id].run_retirement(timetable_idx + 1)

            elif event[0] == 'bus_retirement':

                bus = event[1]
                bus_addition = self.bus_addition_list.pop(0)
                if bus.bus_id == 1:
                    print('retirement at {0}'.format(clk))

                if bus.bus_id == bus_addition[2]:
                    # trajs[bus.bus_id].append((None, None))
                    timetable_idx = bus_addition[3]

                    remove_idx = None
                    for idx, item in enumerate(event_list):
                        if item[0] == 'bus_arrival' or item[0] == 'bus_departure':
                            if item[1].bus_id == bus.bus_id:
                                remove_idx = idx
                                break
                    if remove_idx:
                        del t_list[remove_idx]
                        del event_list[remove_idx]

                    flush_dt = 0
                    # input(bus.timetable)
                    # input(bus_addition)
                    # print(bus.travelled_dist)
                    # print(clk)
                    input(bus.timetable[bus.timetable_idx])
                    for i in range(0, timetable_idx - bus.timetable_idx):
                        # input(bus.timetable[bus.timetable_idx + i])
                        if bus.timetable[bus.timetable_idx + i][1] == 'departure':
                            input('dept')
                            stop_id = bus.timetable[bus.timetable_idx + i][2]
                            # input(stop_id)
                            num_pax_boarding, num_pax_alighting = bus.process_pax(stop_id=stop_id)
                            t_board = num_pax_boarding * bus.pax_board_s  # time for all pax to alight
                            t_alight = num_pax_alighting * bus.pax_alight_s  # time for all pax to board

                            if bus.allow_early:
                                dt = max(t_board, t_alight)
                            else:
                                # note that departure timetable is with respect to ideal case, so no + clk is needed
                                dt = max(t_board, t_alight, bus.timetable[bus.timetable_idx + i][0] - clk)

                        else:
                            input('arr')
                            stop_id = bus.timetable[bus.timetable_idx + i][2]
                            v = np.random.lognormal(bus.mu_speed, bus.sigma_speed)
                            a = np.random.lognormal(bus.mu_acc, bus.sigma_acc)
                            s = bus.stops[stop_id].spacing
                            dt = bus.time_inter_stop(v, a, s)

                        flush_dt += dt
                        # input(clk + flush_dt)
                        dist = bus.timetable[bus.timetable_idx + i][3]
                        # input(dist)
                        trajs[bus.bus_id].append((clk + flush_dt, dist))

                    trajs[bus.bus_id].append((None, None))

                    for i in range(timetable_idx - bus.timetable_idx - 1):
                        schedule = bus.timetable[bus.timetable_idx + i][0]
                        dist_s = bus.timetable[bus.timetable_idx + i][3]
                        sched[bus.bus_id].append((schedule, dist_s))

                    bus.run_retirement(timetable_idx)

        return trajs, self.delays, sched


class Bus:

    def __init__(self, bus_id, headway_s, num_stops, max_clk_s, capacity, mean_speed_kmh, cv_speed,
                 mean_acc_ms2, cv_acc, stops, pax_board_s, pax_alight_s, allow_early, slack_s):
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
        self.allow_early = allow_early
        self.slack_s = slack_s

        # lognormal random variables
        std_speed = self.mean_speed_ms * self.cv_speed
        # obtain mean and std of associated normal distribution for the lognormal distributions of speed using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self.mu_speed = np.log((self.mean_speed_ms ** 2) / np.sqrt(std_speed ** 2 + self.mean_speed_ms ** 2))
        self.sigma_speed = np.sqrt(np.log((std_speed ** 2) / (self.mean_speed_ms ** 2) + 1))

        std_acc = self.mean_acc_ms2 * self.cv_acc
        # obtain mean and std of associated normal distribution for the lognormal distributions of acc rate using
        # mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self.mu_acc = np.log((self.mean_acc_ms2 ** 2) / np.sqrt(std_acc ** 2 + self.mean_acc_ms2 ** 2))
        self.sigma_acc = np.sqrt(np.log((std_acc ** 2) / (self.mean_acc_ms2 ** 2) + 1))

        # build pax storage
        self.pax_lists = {}
        for stop_id in range(self.num_stops):
            self.pax_lists[stop_id] = []
        self.num_pax = 0

        # variables for service state
        self.delay = 0
        self.in_service = False

        # set bus deployment parameters
        # first bus leaves a headway after simulation starts
        self.deploy_clk_s = (self.bus_id + 1) * self.headway_s
        self.stop_id = -1
        self.travelled_dist = - stops[0].spacing

        # build timetable
        self.timetable = self._build_timetable()
        self.timetable_idx = 0

        # schedule first bus arrival
        self.schedule_arrival(abs_t=self.deploy_clk_s)

    def _build_timetable(self):
        """
        this function builds the bus timetable (used for delay measurement)
        :return:
        """

        timetable = []

        # start at first scheduled stop
        dist = 0
        t = self.deploy_clk_s
        stop_id = (self.stop_id + 1) % self.num_stops

        while True:

            if t < self.max_clk_s:

                # time of arrival
                timetable.append((t, 'arrival', stop_id, dist))

                # compute expected time for next departure
                t += max((self.stops[stop_id].pax_s * self.pax_alight_s * self.headway_s),
                         (self.stops[stop_id].pax_s * self.pax_board_s * self.headway_s))

                # add slack
                t += self.slack_s

            if t < self.max_clk_s:
                # time of departure
                timetable.append((t, 'departure', stop_id, dist))

                # compute expected time for next arrival
                v = self.mean_speed_ms
                a = self.mean_acc_ms2
                s = self.stops[stop_id].spacing

                t += self.time_inter_stop(v, a, s)
                stop_id = (stop_id + 1) % self.num_stops
                dist += self.stops[0].spacing
            else:
                break

        return tuple(timetable)

    def schedule_departure(self, num_pax_boarding, num_pax_alighting, abs_t=None):
        """
        this function schedules a bus departure
        :param num_pax_boarding: number of pax boarding bus []
        :param num_pax_alighting: number of pax alighting bus []
        :return:
        """

        global clk
        global t_list
        global event_list

        if (self.in_service or abs_t) and self.timetable_idx < len(self.timetable):
            if not abs_t:
                t_board = num_pax_boarding * self.pax_board_s  # time for all pax to alight
                t_alight = num_pax_alighting * self.pax_alight_s  # time for all pax to board

                if self.allow_early:
                    dt = max(t_board, t_alight)
                else:
                    # note that departure timetable is with respect to ideal case, so no + clk is needed
                    dt = max(t_board, t_alight, self.timetable[self.timetable_idx][0] - clk)
            else:
                self.in_service = True
                # on-time based on timetable
                dt = abs_t - clk

            idx = bisect.bisect_right(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_departure', self))

    def schedule_arrival(self, abs_t=None):
        """
        this function schedules bus stop arrival
        :param first_stop: True if first stop of bus, False is default
        :return:
        """

        global clk
        global t_list
        global event_list

        if (self.in_service or abs_t) and self.timetable_idx < len(self.timetable):
            if not abs_t:
                # random
                v = np.random.lognormal(self.mu_speed, self.sigma_speed)
                a = np.random.lognormal(self.mu_acc, self.sigma_acc)
                s = self.stops[self.stop_id].spacing
                dt = self.time_inter_stop(v, a, s)
            else:
                self.in_service = True
                # on-time based on timetable
                dt = abs_t - clk

            idx = bisect.bisect_right(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_arrival', self))

    def run_departure(self):

        global clk

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

        # schedule next event
        self.timetable_idx += 1
        if self.timetable_idx < len(self.timetable) and self.in_service:
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
        num_pax_boarding, num_pax_alighting = self.process_pax()

        # schedule next event
        self.timetable_idx += 1
        if self.timetable_idx < len(self.timetable) and self.in_service:
            self.schedule_departure(num_pax_boarding, num_pax_alighting)

    def process_pax(self, stop_id=None):
        """
        this function process pax boarding/alighting at a stop
        :return num_pax_boarding: number of pax boarding the bus
        :return num_pax_alighting: number of pax alighting the bus
        """

        if not stop_id:
            stop_id = self.stop_id

        num_pax_alighting = len(self.pax_lists[stop_id])
        self.num_pax -= num_pax_alighting
        self.pax_lists[stop_id] = []

        on_pax = self.stops[stop_id].pax
        num_pax_boarding = 0
        for pax in on_pax:
            if self.num_pax < self.capacity:
                num_pax_boarding += 1
                self.num_pax += 1
                self.pax_lists[pax.destination].append(pax)
            else:
                break

        # remove the boarding passengers from the stop
        self.stops[stop_id].board_pax(num_pax_boarding)

        return num_pax_boarding, num_pax_alighting

    def schedule_retirement(self, abs_t):
        """
        this function schedules a bus retirement
        :param abs_t: absolute simulation time for bus retirement [s]
        :return:
        """
        global t_list
        global event_list

        idx = bisect.bisect_right(t_list, abs_t)
        t_list.insert(idx, abs_t)
        event_list.insert(idx, ('bus_retirement', self))

    def run_retirement(self, timetable_idx):
        """
        this function simply retires a bus
        :return:
        """

        self.in_service = False
        self.timetable_idx = timetable_idx - 1
        self.stop_id = self.timetable[self.timetable_idx][2]
        self.travelled_dist = self.timetable[self.timetable_idx ][3]

        if self.timetable[timetable_idx][1] == 'arrival':
            self.schedule_arrival(abs_t=self.timetable[self.timetable_idx + 1][0])
            # print('arrival at {0}'.format(self.timetable[timetable_idx][0]))
        elif self.timetable[self.timetable_idx][1] == 'departure':
            self.schedule_departure(0, 0, abs_t=self.timetable[self.timetable_idx + 1][0])
            # print('departure at {0}'.format(self.timetable[timetable_idx][0]))

    @staticmethod
    def time_inter_stop(v, a, s):
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
