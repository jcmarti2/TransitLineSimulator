from TransitLineSimulator import TransitLineSimulator
import matplotlib.pyplot as plt
import _pickle as pickle
import operator
import numpy as np
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'


"""
this file runs simulations using the TransitLineSimulator

"""

# ---------------------------------------- #
#                 USER INPUT               #
# ---------------------------------------- #

num_rep = 100                  # number of replications []
max_clk_s = 43200               # maximum simulation time [s]
run_period = 3600               # period for running bus addition strategy [s]
bunch_threshold_s = 312         # bus bunching threshold [s] (default in simulator is False)

num_stops = 24                  # number of stops []
pax_hr = 50                     # pax per hour at each stop [pax/hr]
stop_spacing_m = 1000           # spacing between stops [m]

num_buses = 8                   # number of buses []
headway_s = 432.5                 # headway between buses [s]
bus_capacity = 100              # bus capacity [pax]
bus_mean_speed_kmh = 30         # mean bus cruise speed [km/h]
bus_cv_speed = 0.1             # coefficient of variation for bus speed
bus_mean_acc_ms2 = 1            # mean bus acceleration [m/s2]
bus_cv_acc = 0.1               # coefficient of variation for bus acceleration
pax_board_s = 4                 # boarding time per pax [s/pax]
pax_alight_s = 4                # alighting time per pax [s/pax]

addition_list = [(3272.22, 19000.0, 0, 40), (3396.39, 20000.0, 0, 41), (6668.61, 39000.0, 1, 80), (9892.78, 58000.0, 2, 117)]
# ---------------------------------------- #
#               NO USER INPUT              #
# ---------------------------------------- #

def predict_bunching(delays_dict, threshold):
    """
    inputs are the dictionary with (time,location) as key and [delays] as values
    """
    # create empty dictionary for probabilities
    probaDict = {}

    #for each (time, location) tuple
    for k, delayList in delays_dict.items():
        #retrieve list of delays
        #look for probability for delay to be greater than threshold
        p = sum(i > threshold for i in delayList)/float(len(delayList))
        #create new dictionnary with (time,location) as key and probability as values
        probaDict[k] = p
    #look for largest probability
    insertionTimeLocation = max(probaDict.items(), key = operator.itemgetter(1))[0]
    #return tuple (time, location, stop_id, bus_id)

    input(probaDict[insertionTimeLocation])

    return insertionTimeLocation


def get_greedy_addition():
    # a = -1
    # first simulation time is just run period
    time_to_sim = run_period
    # initially no buses are scheduled for addition
    bus_addition_list = []

    while (time_to_sim - run_period) <= max_clk_s:
        # a += 1
        # compute delays dictionary using all replications
        delays = {}
        for _ in range(num_rep):
            simulator = TransitLineSimulator(time_to_sim, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                             bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                             bus_cv_acc, pax_board_s, pax_alight_s,
                                             delays=delays, bus_addition_list=bus_addition_list.copy())
            _, delays, _ = simulator.simulate()

        # input(delays)
        # with open('/Users/jotaceemeeme/Desktop/GitHub/TransitLineSimulator/output/set_2.cpkl', 'wb+') as file:
        #     pickle.dump(delays, file, protocol=2)
        # set new addition

        # if a == 1:
        #     # input(delays)
        #     for item in delays:
        #         t = item[0]
        #         x = item[1]
        #         plt.scatter(t, x)
        #     plt.show()

        # input(delays)
        # print('time to sim: {0}'.format(time_to_sim))
        new_addition = predict_bunching(delays, bunch_threshold_s)
        bus_addition_list.append(new_addition)
        input(bus_addition_list)
        # input(bus_addition_list)
        #
        # new batch of replications will start at bus addition time
        time_to_sim = new_addition[0] + run_period

    return bus_addition_list

def simulate_line(addition_list=None, bunch_threshold_s=False):

    simulator = TransitLineSimulator(10000, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                     bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                     bus_cv_acc, pax_board_s, pax_alight_s,
                                     delays=None, bus_addition_list=addition_list, bunch_threshold_s=bunch_threshold_s)

    traj, delays, sched = simulator.simulate()

    return traj, delays, sched


traj, delays, sched = simulate_line(bunch_threshold_s=bunch_threshold_s)
ct = 0
for bus in traj:
    time, dist = zip(*traj[bus])
    schedule, dist = zip(*sched[bus])
    if ct == 0:
        plt.plot(time, dist, 'k', label='Trajectory')
        plt.plot(schedule, dist, 'b', linestyle='--', label='Schedule')
        ct += 1
    else:
        plt.plot(time, dist, 'k')
        plt.plot(schedule, dist, 'b', linestyle='--')
plt.legend(loc=2)
plt.xlabel('Time [s]')
plt.ylabel('Distance [m]')
plt.show()

# get_greedy_addition()