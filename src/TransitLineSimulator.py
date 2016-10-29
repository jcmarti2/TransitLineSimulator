import bisect
import numpy as np
from collections import namedtuple

__author__ = 'juan carlos martinez mori'

clk = 0.0
t_list = []
event_list = []


class TransitLineSimulator:

    def __init__(self, max_clk_s=3600, headway_s=300, num_stops=10, pax_hr=20, stop_spacing_m=1000, num_buses=8,
                 bus_capacity=30, bus_mean_speed_kmh=30, bus_cv_speed=0.01, bus_mean_acc_ms2=5, bus_cv_acc=0.01,
                 pax_board_s=4, pax_alight_s=4):

        self._max_clk_s = max_clk_s
        self._headway_s = headway_s

        self._num_stops = num_stops
        self._pax_hr = pax_hr
        self._stop_spacing_m = stop_spacing_m

        self._num_buses = num_buses
        self._bus_capacity = bus_capacity
        self._bus_mean_speed_kmh = bus_mean_speed_kmh
        self._bus_cv_speed = bus_cv_speed
        self._bus_mean_acc_ms2 = bus_mean_acc_ms2
        self._bus_cv_acc = bus_cv_acc

        self._pax_board_s = pax_board_s
        self._pax_alight_s = pax_alight_s

        self.stops = []
        self._build_stops()
        self.buses = []
        self._build_buses()

    def _build_stops(self):

        for stop_id in range(self._num_stops):
            spacing = self._stop_spacing_m
            other_stop_ids = list(range(self._num_stops))
            del other_stop_ids[stop_id]
            self.stops.append(Stop(stop_id, self._pax_hr, spacing, other_stop_ids))

    def _build_buses(self):

        for bus_id in range(self._num_buses):
            self.buses.append(Bus(bus_id, self._headway_s, self._num_stops, self._max_clk_s, self._bus_capacity,
                                  self._bus_mean_speed_kmh, self._bus_cv_speed, self._bus_mean_acc_ms2,
                                  self._bus_cv_acc, self.stops, self._pax_board_s, self._pax_alight_s))

    def _initialize(self):

        global t_list
        global event_list

        warmup_timetable = self.buses[0].timetable  # first bus needs to cover all stops
        stop_id = 0
        for event in warmup_timetable:
            if event[1] == 'arrival' and stop_id < self._num_stops:
                t = event[0]
                idx = bisect.bisect_left(t_list, t)  # bisect left because pax arrival has priority
                t_list.insert(idx, t)
                event_list.insert(idx, ('pax_arrival', self.stops[stop_id]))
                stop_id += 1
            else:
                break

    def simulate(self):

        global clk
        global t_list
        global event_list

        self._initialize()
        while t_list and event_list:

            if t_list[0] > self._max_clk_s:
                break

            clk = t_list.pop(0)
            event = event_list.pop(0)

            if event[0] == 'bus_arrival':
                bus = event[1]
                bus.run_arrival()
            elif event[0] == 'bus_departure':
                bus = event[1]
                bus.run_departure()
            else:
                stop = event[1]
                stop.run_pax_arrival()

class Bus:

    def __init__(self, bus_id, headway_s, num_stops, max_clk_s, capacity, mean_speed_kmh, cv_speed,
                 mean_acc_ms2, cv_acc, stops, pax_board_s, pax_alight_s, init_stop=0, add_bus_s=None):

        self.bus_id = bus_id
        self.headway_s = headway_s
        self.num_stops = num_stops
        self.max_clk_s = max_clk_s
        self.capacity = capacity

        self.mean_speed_ms = mean_speed_kmh / 3.6
        self.cv_speed = cv_speed
        self.mean_acc_ms2 = mean_acc_ms2
        self.cv_acc = cv_acc

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

        self.stops = stops
        self.pax_board_s = pax_board_s
        self.pax_alight_s = pax_alight_s
        self.stop_id = init_stop
        if not add_bus_s:
            self._deploy_clk_s = self.bus_id * self.headway_s
        else:
            self._deploy_clk_s = add_bus_s

        self.timetable = self._build_timetable()
        self.timetable_idx = 0
        self.delay = 0

        self.pax_lists = {}
        for stop_id in range(self.num_stops):
            self.pax_lists[stop_id] = []
        self.num_pax = 0

        self.in_service = False
        self.schedule_arrival(first_stop=True)

    def _build_timetable(self):

        timetable = []

        t = self._deploy_clk_s
        stop_id = self.stop_id % self.num_stops

        while True:

            if t < self.max_clk_s:
                timetable.append((t, 'arrival', stop_id))
                t += max((self.stops[stop_id].pax_s * self.pax_alight_s * self.headway_s),
                         (self.stops[stop_id].pax_s * self.pax_board_s * self.headway_s))

            elif t < self.max_clk_s:
                timetable.append((t, 'departure', stop_id))

                v = self.mean_speed_ms
                a = self.mean_acc_ms2
                s = self.stops[stop_id].spacing

                t += self._time_inter_stop(v, a, s)
                stop_id = (stop_id + 1) % self.num_stops

            else:
                break

        return tuple(timetable)

    def schedule_departure(self, num_pax_boarding, num_pax_alighting):

        global clk
        global t_list
        global event_list

        if self.in_service:
            t_board = num_pax_boarding * self.pax_board_s  # time for all pax to alight
            t_alight = num_pax_alighting * self.pax_alight_s  # time for all pax to board

            # note that departure timetable is with respect to ideal case, so no + clk is needed
            t = max(t_board, t_alight, self.timetable[self.timetable_idx][0] - clk)

            idx = bisect.bisect_left(t_list, clk + t)
            t_list.insert(idx, clk + t)
            event_list.insert(idx, ('bus_departure', self))

    def schedule_arrival(self, first_stop=False):

        global clk
        global t_list
        global event_list

        if self.in_service or first_stop:
            if not first_stop:
                v = np.random.lognormal(self._mu_speed, self._sigma_speed)
                a = np.random.lognormal(self._mu_acc, self._sigma_acc)
                s = self.stops[self.stop_id].spacing
                dt = self._time_inter_stop(v, a, s)
            else:
                dt = self.timetable[self.timetable_idx][0] - clk

            idx = bisect.bisect_left(t_list, clk + dt)
            t_list.insert(idx, clk + dt)
            event_list.insert(idx, ('bus_arrival', self))

    def run_departure(self):

        global clk

        # update delay
        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])
        self.schedule_arrival()
        self.timetable_idx += 1

    def run_arrival(self):

        global clk

        if self.timetable_idx == 0:
            self.in_service = True

        self.delay = max(0, clk - self.timetable[self.timetable_idx][0])

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
        self.schedule_departure(num_pax_alighting, num_pax_boarding)
        self.timetable_idx += 1

    @staticmethod
    def _time_inter_stop(v, a, s):

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