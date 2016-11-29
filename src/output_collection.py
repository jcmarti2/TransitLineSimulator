from TransitLineSimulator import TransitLineSimulator
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'

num_rep = 100  # number of replications []
max_clk_s = 14400  # maximum simulation time [s]
run_period = 900  # period for running bus addition strategy [s]
bunch_threshold_s = 312  # bus bunching threshold [s] (default in simulator is False)
p_threshold = 0.3  # probability threshold for bus addition
allow_early = False  # True if allowing for earliness, False otherwise
slack_s = 0  # slack per stop [s]

num_stops = 24  # number of stops []
pax_hr = 50  # pax per hour at each stop [pax/hr]
stop_spacing_m = 1000  # spacing between stops [m]

num_buses = 8  # number of buses []
headway_s = 432.5  # headway between buses [s]
bus_capacity = 100  # bus capacity [pax]
bus_mean_speed_kmh = 30  # mean bus cruise speed [km/h]
bus_cv_speed = 0.1  # coefficient of variation for bus speed
bus_mean_acc_ms2 = 1  # mean bus acceleration [m/s2]
bus_cv_acc = 0.1  # coefficient of variation for bus acceleration
pax_board_s = 4  # boarding time per pax [s/pax]
pax_alight_s = 4  # alighting time per pax [s/pax]


def simulate_no_insertion(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                          bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                          allow_early, slack_s, plot_traj=False):
    mode = 'no_insertion'
    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                                     bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s,
                                     pax_alight_s, allow_early, slack_s, mode)

    bus_records, _ = simulator.simulate()

    if plot_traj:
        plot_trajectories(bus_records)


def plot_trajectories(bus_records):

    plt.figure()
    for bus_id in bus_records:
        bus_record = list(zip(*bus_records[bus_id]))
        clk = bus_record[0]
        abs_dist = bus_record[1]
        schedule_t = bus_record[2]
        schedule_dist = bus_record[3]
        plt.plot(clk, abs_dist)
        plt.plot(schedule_t, schedule_dist, linestyle='--')
    plt.show()


if __name__ == '__main__':
    simulate_no_insertion(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                          bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                          allow_early, slack_s, plot_traj=True)
