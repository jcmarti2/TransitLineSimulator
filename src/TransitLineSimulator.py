import bisect
import numpy as np
from collections import namedtuple

__author__ = 'juan carlos martinez mori'

clk = 0.0
t_list = []
event_list = []


class TransitLineSimulator:

    def __init__(self, max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                 bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                 allow_early, slack_s, mode, bunch_threshold_s=False, bus_addition_list=None, delay_start_s=0):
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
        :param bunch_threshold_s: threshold for bus addition
        :param bus_addition_list: list of scheduled bus additions (
        :param delay_start_s: start time for storing delay data [s]
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

        self.mode = mode

        # TODO: make sure required parameters are passed
        if self.mode == 'no_insertion':
            pass
        elif self.mode == 'reactive':
            pass
        elif self.mode == 'preventive':
            pass
        else:
            raise Exception('Mode not supported. Select from no_insertion, reactive or preventive')

        self.delays = {}
        self.bus_records = {}
        self.bunch_threshold_s = bunch_threshold_s
        self.bus_addition_list = bus_addition_list
        self.delay_start_s = delay_start_s

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
            run_id = bus_id

            # first bus departs one headway after simulation starts
            deploy_clk_s = (bus_id + 1) * self.headway_s

            # buses start at first stop with abs_dist of 0
            deploy_stop_id = 0
            deploy_abs_dist = 0

            # add bus to dictionary
            self.buses[bus_id] = (Bus(bus_id, run_id, deploy_clk_s, deploy_stop_id, deploy_abs_dist, self.headway_s,
                                      self.num_stops, self.max_clk_s, self.bus_capacity, self.bus_mean_speed_kmh,
                                      self.bus_cv_speed, self.bus_mean_acc_ms2, self.bus_cv_acc, self.stops,
                                      self.pax_board_s, self.pax_alight_s, self.allow_early, self.slack_s))

        # build added buses
        if self.bus_addition_list:
            for addition in self.bus_addition_list:
                self._process_added_bus(addition)

    def _process_added_bus(self, addition):
        """
        this function process a bus addition
        :param record: entry from self.bus_addition_list (schedule_t, schedule_dist, schedule_stop_id, retired_bus_id)
        :return:
        """

        # extract information for bus being added
        deploy_clk_s = addition[0]
        deploy_abs_dist = addition[1]
        deploy_stop_id = addition[2]
        retired_bus_id = addition[3]

        # assign ids for bus being added
        run_id = self.buses[retired_bus_id].run_id
        bus_id = self.num_buses
        self.num_buses += 1

        # add bus to dictionary
        self.buses[bus_id] = (Bus(bus_id, run_id, deploy_clk_s, deploy_stop_id, deploy_abs_dist, self.headway_s,
                                  self.num_stops, self.max_clk_s, self.bus_capacity, self.bus_mean_speed_kmh,
                                  self.bus_cv_speed, self.bus_mean_acc_ms2, self.bus_cv_acc, self.stops,
                                  self.pax_board_s, self.pax_alight_s, self.allow_early, self.slack_s))

        # set retire absolute distance for retiring bus
        self.buses[retired_bus_id].retire_abs_dist = deploy_abs_dist

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

        # set dictionary to store trajectory and timetable data
        for bus_id in self.buses:
            # format for tuples in list is (clk, abs_dist, scheduled_t, scheduled_dist)
            self.bus_records[bus_id] = []

        # while there are scheduled events
        while t_list and event_list:

            # terminate if max simulation time is exceeded
            if t_list[0] > self.max_clk_s:
                break

            # get event
            clk = t_list.pop(0)
            event = event_list.pop(0)

            # run passenger arrival event
            if event[0] == 'pax_arrival':
                stop = event[1]
                stop.run_pax_arrival()

            elif event[0] == 'bus_arrival' or event[0] == 'bus_departure':

                bus = event[1]
                if bus.timetable_idx < len(bus.timetable) and bus.in_service:
                    self._process_bus_event(bus, event[0])

        return self.bus_records, self.delays

    def _process_bus_event(self, bus, event_type):
        """
        process bus event
        :param bus: bus object
        :param event_type: 'bus_arrival' or 'bus_departure'
        :return:
        """

        if event_type == 'bus_arrival':
            bus.run_arrival()
        elif event_type == 'bus_departure':
            bus.run_departure()

        if bus.timetable_idx < len(bus.timetable):
            record = self._get_bus_record(bus)
            self.bus_records[bus.bus_id].append(record)

            # build tuple for bus addition
            addition = tuple([record[2], record[3], bus.timetable[bus.timetable_idx][2], bus.bus_id])

            # process addition strategy
            if self.mode == 'preventive':

                if record[2] > self.delay_start_s:
                    self.delays.setdefault(addition, default=[]).append(bus.delay)

            elif self.mode == 'reactive' and event_type == 'bus_departure':

                if bus.delay >= self.bunch_threshold_s and not bus.retire_abs_dist:
                    self._process_added_bus(addition)

    @staticmethod
    def _get_bus_record(bus):
        """
        this function formats a bus trajectory record
        :param bus: bus object
        :return record: tuple in format (clk, abs_dist, scheduled_t, scheduled_dist)
        """

        global clk

        schedule_t = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx][0]))
        schedule_dist = float('{0:.2f}'.format(bus.timetable[bus.timetable_idx][3]))
        abs_dist = float('{0:.2f}'.format(bus.abs_dist))

        record = tuple([clk, abs_dist, schedule_t, schedule_dist])

        return record


class Bus:

    def __init__(self, bus_id, run_id, deploy_clk_s, deploy_stop_id, deploy_abs_dist, headway_s, num_stops, max_clk_s,
                 capacity, mean_speed_kmh, cv_speed, mean_acc_ms2, cv_acc, stops, pax_board_s, pax_alight_s,
                 allow_early, slack_s, retire_abs_dist=None):
        """
        constructor for the buss class
        :param bus_id: bus unique id as an integer []
        :param run_id: run unique id as an integer []
        :param deploy_clk_s: deploy time as an integer [s]
        :param deploy_stop_id: deploy stop_id as an integer []
        :param deploy_abs_dist: deploy absolute distance as float [m]
        :param headway_s: headway between buses [s]
        :param capacity: bus capacity [pax]
        :param mean_speed_kmh: mean bus cruise speed [km/h]
        :param cv_speed: coefficient of variation for bus speed []
        :param mean_acc_ms2: mean bus acceleration [m/s2]
        :param cv_acc: coefficient of variation for bus acceleration []
        :param pax_board_s: boarding time per pax [s/pax]
        :param pax_alight_s: alighting time per pax [s/pax]
        :param allow_early: True if allow bus run early, False to stick to schedule []
        :param slack_s: time slack at each stop as float [s]
        :param retire_abs_dist: absolute distance at which retirement occurs [m]
        """

        # save input
        self.bus_id = bus_id
        self.run_id = run_id
        self.deploy_clk_s = deploy_clk_s
        self.deploy_stop_id = deploy_stop_id
        self.deploy_abs_dist = deploy_abs_dist
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
        self.retire_abs_dist = retire_abs_dist

        # compute lognormal distribution parameters
        self.mu_speed, self.sigma_speed = self.get_lognormal_mu_sigma(self.mean_speed_ms, self.cv_speed)
        self.mu_acc, self.sigma_acc = self.get_lognormal_mu_sigma(self.mean_acc_ms2, self.cv_acc)

        # build pax storage
        self.pax_lists = {}
        for stop_id in range(self.num_stops):
            self.pax_lists[stop_id] = []
        self.num_pax = 0

        # variables for service state
        self.delay = 0
        self.in_service = False

        # set bus deployment parameters
        self.stop_id = deploy_stop_id - 1
        self.abs_dist = self.deploy_abs_dist - stops[self.deploy_stop_id].spacing

        # build timetable
        self.timetable = self.build_timetable()
        self.timetable_idx = -1

        # schedule first bus arrival
        self.schedule_arrival(abs_t=self.deploy_clk_s)

    def build_timetable(self):
        """
        this function builds the bus timetable (used for delay measurement)
        :return timetable: tuple containing timetable in format [(time, 'arrival' or 'departure', stop_id, dist), ...]
        """

        timetable = []

        # start at first scheduled stop
        t = self.deploy_clk_s
        dist = self.deploy_abs_dist
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

    def schedule_departure(self, num_pax_boarding, num_pax_alighting):
        """
        this function schedules a bus departure
        :param num_pax_boarding: number of pax boarding bus []
        :param num_pax_alighting: number of pax alighting bus []
        :param abs_t: absolute time for bus stop departure as a float [s]
        :return:
        """

        global clk
        global t_list
        global event_list

        if (self.in_service or abs_t) and self.timetable_idx < len(self.timetable):

            # random
            t_board = num_pax_boarding * self.pax_board_s  # time for all pax to alight
            t_alight = num_pax_alighting * self.pax_alight_s  # time for all pax to board

            if self.allow_early:
                dt = max(t_board, t_alight)
            else:
                # note that departure timetable is with respect to ideal case, so no + clk is needed
                dt = max(t_board, t_alight, self.timetable[self.timetable_idx][0] - clk)

            idx = bisect.bisect_right(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_departure', self))

    def schedule_arrival(self, abs_t=None):
        """
        this function schedules bus stop arrival
        :param abs_t: absolute time for bus stop arrival as a float [s]
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
                # based on abs_t
                self.in_service = True
                dt = abs_t - clk

            idx = bisect.bisect_right(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_arrival', self))

    def run_departure(self):
        """
        this function runs a bus departure from a stop
        :return:
        """

        global clk

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

        # update timetable index at end of event
        self.timetable_idx += 1

        # schedule next bus arrival
        if self.timetable_idx < len(self.timetable) and self.in_service:
            self.schedule_arrival()

    def run_arrival(self):
        """
        this function run a bus arrival to a stop
        :return:
        """

        global clk

        # new stop, so increase stop id and absolute distance
        self.stop_id = (self.stop_id + 1) % self.num_stops
        self.abs_dist += self.stops[self.stop_id].spacing

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

        # run pax processing
        num_pax_boarding, num_pax_alighting = self.process_pax()

        # update timetable index at end of event
        self.timetable_idx += 1

        # schedule next event
        if self.timetable_idx < len(self.timetable) and self.in_service:
            self.schedule_departure(num_pax_boarding, num_pax_alighting)

    def process_pax(self):
        """
        this function process pax boarding/alighting at a stop
        :return num_pax_boarding: number of pax boarding the bus as integer []
        :return num_pax_alighting: number of pax alighting the bus as integer []
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

    def retire(self):
        """
        this function simply retires a bus
        :return:
        """
        self.in_service = False

    @staticmethod
    def get_lognormal_mu_sigma(mean, cv):
        """
        this function converts a mean and coefficient of variation to mu and sigma for lognormal distribution
        according to mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        :param mean: mean as float
        :param cv: coefficient of variation as float
        :return mu: as float
        :return sigma: as float
        """
        std = mean * cv
        mu = np.log((mean ** 2) / np.sqrt(std ** 2 + mean ** 2))
        sigma = np.sqrt(np.log((std ** 2) / (mean ** 2) + 1))
        return mu, sigma

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
