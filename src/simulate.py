from TransitLineSimulator import TransitLineSimulator

__author__ = 'juan carlos martinez mori'


"""
this file runs simulations using the TransitLineSimulator

"""

# ---------------------------------------- #
#                 USER INPUT               #
# ---------------------------------------- #

num_rep = 1000                  # number of replications []
max_clk_s = 43200               # maximum simulation time [s]
run_period = 3600               # period for running bus addition strategy [s]
bunch_threshold_s = 100         # bus bunching threshold [s] (default in simulator is False)

num_stops = 24                  # number of stops []
pax_hr = 50                     # pax per hour at each stop [pax/hr]
stop_spacing_m = 1000           # spacing between stops [m]

num_buses = 8                   # number of buses []
headway_s = 360                 # headway between buses [s]
bus_capacity = 100              # bus capacity [pax]
bus_mean_speed_kmh = 30         # mean bus cruise speed [km/h]
bus_cv_speed = 0.01             # coefficient of variation for bus speed
bus_mean_acc_ms2 = 1            # mean bus acceleration [m/s2]
bus_cv_acc = 0.01               # coefficient of variation for bus acceleration
pax_board_s = 4                 # boarding time per pax [s/pax]
pax_alight_s = 4                # alighting time per pax [s/pax]


# ---------------------------------------- #
#               NO USER INPUT              #
# ---------------------------------------- #

def manage_bus_addition(delay):
    return -1


def get_greedy_addition():

    # first simulation time is just run period
    time_to_sim = run_period
    # initially no buses are scheduled for addition
    bus_addition_list = []

    while (time_to_sim - run_period) <= max_clk_s:

        # compute delays dictionary using all replications
        delays = {}
        for _ in range(num_rep):
            simulator = TransitLineSimulator(time_to_sim, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                             bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                             bus_cv_acc, pax_board_s, pax_alight_s,
                                             delays=delays, bus_addition_list=bus_addition_list)
            _, delays = simulator.simulate()

        # set new addition
        new_addition = manage_bus_addition(delays)
        bus_addition_list.append(new_addition)

        # new batch of replications will start at bus addition time
        time_to_sim = new_addition[0] + run_period

    return bus_addition_list


def simulate_line(addition_list=None, bunch_threshold_s=False):

    simulator = TransitLineSimulator()
    traj, _ = simulator.simulate(max_clk_s, addition_list)

    return traj


