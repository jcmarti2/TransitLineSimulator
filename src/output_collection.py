from TransitLineSimulator import TransitLineSimulator
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'

num_rep = 100  # number of replications []
max_clk_s = 14400  # maximum simulation time [s]
run_period = 900  # period for running bus addition strategy [s]
bunch_threshold_s = 312  # bus bunching threshold [s] (default in simulator is False)
p_threshold = 0.5  # probability threshold for bus addition
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
        plot_trajectories(bus_records, 'no')


def simulate_reactive(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                      bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                      allow_early, slack_s, bunch_threshold_s, plot_traj=False):
    mode = 'reactive'
    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                                     bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s,
                                     pax_alight_s, allow_early, slack_s, mode, bunch_threshold_s=bunch_threshold_s)

    bus_records, _ = simulator.simulate()

    if plot_traj:
        plot_trajectories(bus_records, mode)


def simulate_preventive(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                        bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                        allow_early, slack_s, run_period, num_rep, bunch_threshold_s, p_threshold,
                        bus_addition_list=None, plot_traj=False):

    delay_start_s = 0
    mode = 'preventive'
    bus_addition_list = [(3248.19, 19000.0, 20, 0), (5162.64, 29000.0, 6, 1), (6187.92, 33000.0, 10, 2), (7361.39, 38000.0, 15, 3), (7485.42, 33000.0, 10, 5), (7497.5, 36000.0, 13, 4), (8646.81, 35000.0, 12, 7), (9251.67, 42000.0, 19, 6), (10213.33, 66000.0, 20, 8), (10497.64, 65000.0, 19, 9), (11671.11, 70000.0, 0, 10), (11795.14, 65000.0, 19, 13), (12696.39, 74000.0, 4, 11), (12944.45, 64000.0, 18, 14), (13413.2, 73000.0, 3, 12)]

    if not bus_addition_list:

        cum_time = 0
        bus_addition_list = []
        p_list = []
        inserted_bus_ids = set()

        while max_clk_s - cum_time > run_period:

            cum_delays = {}
            sim_time = cum_time + run_period

            for _ in range(num_rep):
                simulator = TransitLineSimulator(sim_time, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                                 bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                                 bus_cv_acc, pax_board_s, pax_alight_s, allow_early, slack_s, mode,
                                                 bus_addition_list=bus_addition_list, delay_start_s=delay_start_s)
                _, delays = simulator.simulate()

                for key in delays:
                    cum_delays.setdefault(key, []).append(delays[key])

            new_addition, p, inserted_bus_ids = predict_bunching(cum_delays, bunch_threshold_s, p_threshold,
                                                                 inserted_bus_ids)
            print(bus_addition_list)
            if new_addition:
                bus_addition_list.append(new_addition)
                p_list.append(p)
                cum_time = new_addition[0]
            else:
                cum_time += run_period

            delay_start_s = cum_time

    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                     bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                     bus_cv_acc, pax_board_s, pax_alight_s, allow_early, slack_s, mode,
                                     bus_addition_list=bus_addition_list, delay_start_s=delay_start_s)
    bus_records, _ = simulator.simulate()

    if plot_traj:
        plot_trajectories(bus_records, mode)


def predict_bunching(cum_delays, bunch_threshold_s, p_threshold, inserted_bus_ids):

    candidates = []
    for key in cum_delays:

        delay_list = cum_delays[key]

        p = sum(i > bunch_threshold_s for i in delay_list) / len(delay_list)

        if p >= p_threshold:
            candidates.append(tuple([p, key]))

    candidates.sort(key=lambda x: x[0])
    for candidate in candidates:
        if candidate[1][-1] not in inserted_bus_ids:
            inserted_bus_ids.add(candidate[1][-1])
            return candidate[1], candidate[0], inserted_bus_ids

    return None, None, inserted_bus_ids


def plot_trajectories(bus_records, mode):

    plt.figure()
    sum_x = 0
    sum_t = 0

    labeled = False

    for bus_id in bus_records:
        bus_record = list(zip(*bus_records[bus_id]))
        clk = bus_record[0]
        abs_dist = bus_record[1]
        schedule_t = bus_record[2]
        schedule_dist = bus_record[3]
        if not labeled:
            plt.plot(clk, abs_dist, 'k', label='Trajectory')
            plt.plot(schedule_t, schedule_dist, 'b', linestyle='--', label='Schedule')
            labeled = True
        else:
            plt.plot(clk, abs_dist, 'k')
            plt.plot(schedule_t, schedule_dist, 'b', linestyle='--')

        delta_x = abs_dist[-1] - abs_dist[0]
        delta_t = clk[-1] - clk[0]

        sum_x += delta_x
        sum_t += delta_t

    mean_speed = (sum_x / sum_t) * 3.6
    plt.legend(loc=2)
    plt.title('Strategy: {0} Insertion \n Mean Commercial Speed: {1:.2f} km/h'.format(mode.title(), mean_speed))
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.savefig('/Users/jotaceemeeme/Desktop/GitHub/TransitLineSimulator/output/ani_1/traj.png', dpi=600)
    compile_animation(bus_records, mode, mean_speed, num_stops, stop_spacing_m)

    # plt.show()


def compile_animation(bus_records, mode, mean_speed, num_stops, stop_spacing_m):

    r = 1
    angles = np.linspace(0, 2 * np.pi, num_stops)
    locs = [[r * np.cos(angle), r * np.sin(angle)] for angle in angles]
    x, y = zip(*locs)

    loc_dict = {}
    for idx, loc in enumerate(locs):
        loc_dict[idx] = loc

    fig, ax = plt.subplots()
    route = plt.Circle((0, 0), 1, color='k', fill=False)

    frames = []
    for bus_id in bus_records:
        bus_record = bus_records[bus_id]
        for idx, loc in enumerate(bus_record):
            clk = loc[0]
            abs_dist = loc[1]
            loc_dict_idx = (int(abs_dist) % (num_stops * stop_spacing_m)) / stop_spacing_m

            if idx == len(bus_record) - 1:
                frames.append([bus_id, clk, loc_dict_idx, -1])
            else:
                frames.append([bus_id, clk, loc_dict_idx])
    frames.sort(key=lambda step: step[1])

    path = '/Users/jotaceemeeme/Desktop/GitHub/TransitLineSimulator/output/ani_1'

    cur_locs = {}
    for idx, frame in enumerate(frames):

        cur_locs[frame[0]] = frame[2]
        clk = frame[1]
        ax.cla()

        for bus_id in cur_locs:
            px, py = loc_dict[cur_locs[bus_id]]
            ax.scatter(px, py, marker='s', color='g', s=100)

        ax.scatter(x, y, color='k', marker='o')
        ax.add_artist(route)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0, -0.3, 'Strategy: {0} Insertion \n Mean Commercial Speed: {1:.2f} km/h \n '
                         'Number of Buses In Line: {2} \n Clock Time: {3:.2f} s'.format(mode.title(), mean_speed,
                                                                                        len(cur_locs), clk),
                fontsize=15, horizontalalignment='center')

        fig.savefig('{0}/{1}.png'.format(path, idx), dpi=300)

        if frame[-1] == -1:
            del cur_locs[frame[0]]


if __name__ == '__main__':
    # simulate_no_insertion(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
    #                       bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
    #                       allow_early, slack_s, plot_traj=True)

    # simulate_reactive(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
    #                   bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
    #                   allow_early, slack_s, bunch_threshold_s, plot_traj=True)

    simulate_preventive(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                        bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s, pax_alight_s,
                        allow_early, slack_s, run_period, num_rep, bunch_threshold_s, p_threshold, plot_traj=True)

