from TransitLineSimulator import TransitLineSimulator
import matplotlib.pyplot as plt
import _pickle as pickle
import operator
import numpy as np
from matplotlib import rc
import itertools
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'


"""
this file runs simulations using the TransitLineSimulator

"""

# ---------------------------------------- #
#               NO USER INPUT              #
# ---------------------------------------- #


def predict_bunching(delays_dict, delay_threshold, p_threshold, last_addition):
    """
    inputs are the dictionary with (time,location) as key and [delays] as values
    :param delays_dict: dictionary of delays
    :param delay_threshold: time delay threshold for bus addition
    :param p_threshold: probability threshold for bus addition
    :return:
    """

    # create empty dictionary for probabilities
    prob_dict = {}

    # for each (time, location) tuple
    for key, delay_list in delays_dict.items():

        # retrieve list of delays and look for probability for delay to be greater than threshold
        p = sum(i > delay_threshold for i in delay_list)/float(len(delay_list))

        # create new dictionary with (time,location) as key and probability as values
        prob_dict[key] = p

    # look for largest probability
    max_p = 0
    insertion_key = None
    # input(last_addition)
    for key, value in prob_dict.items():
        # input(key)
        # input(value)
        if value > p_threshold:
            if not last_addition:
                insertion_key = key
            elif (
                    (value > max_p) and
                    (key[2] != last_addition[2]) or
                    (key[2] == last_addition[2] and last_addition[3] != key[3] + 1)
            ):
                insertion_key = key
                max_p = value

    return insertion_key


def get_greedy_addition():

    # first simulation time is just run period
    time_to_sim = run_period
    # initially no buses are scheduled for addition
    bus_addition_list = []

    last_addition = None
    while (time_to_sim - run_period) <= max_clk_s:
        # a += 1
        # compute delays dictionary using all replications
        delays = {}
        for _ in range(num_rep):
            simulator = TransitLineSimulator(time_to_sim, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                             bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                             bus_cv_acc, pax_board_s, pax_alight_s, allow_early,
                                             delays=delays, bus_addition_list=bus_addition_list.copy())
            _, delays, _ = simulator.simulate()

        new_addition = predict_bunching(delays, bunch_threshold_s, p_threshold, last_addition)
        if new_addition:
            bus_addition_list.append(new_addition)

            # new batch of replications will start at bus addition time
            time_to_sim = new_addition[0] + run_period
        else:
            time_to_sim += run_period

        last_addition = new_addition
        print('Addition list: {0}'.format(bus_addition_list))

    return bus_addition_list

def isplit(iterable,splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

def simulate_line(addition_list=None, bunch_threshold_s=False):

    simulator = TransitLineSimulator(32000, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                     bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                     bus_cv_acc, pax_board_s, pax_alight_s, allow_early,
                                     delays=None, bus_addition_list=addition_list, bunch_threshold_s=bunch_threshold_s)

    traj, delays, sched = simulator.simulate()

    return traj, delays, sched


# ---------------------------------------- #
#                 USER INPUT               #
# ---------------------------------------- #

if __name__ == '__main__':

    num_rep = 100                   # number of replications []
    max_clk_s = 43200               # maximum simulation time [s]
    run_period = 900                # period for running bus addition strategy [s]
    bunch_threshold_s = 312         # bus bunching threshold [s] (default in simulator is False)
    p_threshold = 0.3               # probability threshold for bus addition
    allow_early = False              # True if allowing for earliness, False otherwise

    num_stops = 24                  # number of stops []
    pax_hr = 50                     # pax per hour at each stop [pax/hr]
    stop_spacing_m = 1000           # spacing between stops [m]

    num_buses = 8                   # number of buses []
    headway_s = 432.5               # headway between buses [s]
    bus_capacity = 100              # bus capacity [pax]
    bus_mean_speed_kmh = 30         # mean bus cruise speed [km/h]
    bus_cv_speed = 0.1              # coefficient of variation for bus speed
    bus_mean_acc_ms2 = 1            # mean bus acceleration [m/s2]
    bus_cv_acc = 0.1                # coefficient of variation for bus acceleration
    pax_board_s = 4                 # boarding time per pax [s/pax]
    pax_alight_s = 4                # alighting time per pax [s/pax]

    addition_list = [(4309.58, 26000.0, 0, 54), (5346.94, 33000.0, 0, 68), (5779.44, 33000.0, 1, 68), (5927.64, 34000.0, 1, 70), (6360.14, 34000.0, 2, 70), (6656.53, 36000.0, 2, 74), (6940.83, 35000.0, 3, 72), (7201.11, 34000.0, 4, 69)]

    traj, delays, sched = simulate_line(bunch_threshold_s=bunch_threshold_s)
    ct_t = 0
    ct_s = 0
    total_dist = 0
    total_time = 0

    for bus in traj:
        time_l, dist_l = zip(*traj[bus])
        time_l = isplit(time_l, (None,))
        dist_l = isplit(dist_l, (None,))
        schedule, dist_s = zip(*sched[bus])
        if ct_s == 0:
            plt.plot(schedule, dist_s, 'b', linestyle='--', label='Schedule')
            ct_s += 1
        else:
            plt.plot(schedule, dist_s, 'b', linestyle='--')
        for time, dist in zip(time_l, dist_l):
            if ct_t == 0:
                plt.plot(time, dist, 'k', label='Trajectory')
                total_dist += dist[-1] - dist[0]
                total_time += time[-1] - time[0]
                ct_t += 1
            else:
                plt.plot(time, dist, 'k')
                delta_dist = dist[-1] - dist[0]
                delta_time = time[-1] - time[0]
                total_dist += (dist[-1] - dist[0])
                total_time += (time[-1] - time[0])

    # v = (total_dist / total_time) * 3.6

    # input(v)
    #
    #     if ct == 0:
    #         plt.plot(time, dist, 'k', label='Trajectory')
    #         plt.plot(schedule, dist, 'b', linestyle='--', label='Schedule')
    #         ct += 1
    #     else:
    #         plt.plot(time, dist, 'k')
    #         plt.plot(schedule, dist, 'b', linestyle='--')
    plt.legend(loc=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.show()
    #
    # get_greedy_addition()