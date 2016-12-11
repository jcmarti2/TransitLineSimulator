from src.TransitLineSimulator import TransitLineSimulator
from src.sim_parameters import *
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import pickle
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'
path = os.path.dirname(os.path.abspath(__file__))


def simulate_no_insertion(save_output=False, rep_id=None, plot_traj=False, save_fig=False, animate=False):

    mode = 'no_insertion'
    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                                     bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s,
                                     pax_alight_s, allow_early, slack_s, mode)

    bus_records, _ = simulator.simulate()
    metric = compute_metric(bus_records)

    allow_early_msg = 'Allow Earliness' if allow_early else 'No Earliness'
    slack_msg = 'Slack' if slack_s else 'No Slack'
    strategy = 'No Insertion, {0}, {1}'.format(allow_early_msg, slack_msg)

    output = {'bus_records': bus_records, 'bus_addition_list': [], 'metric': metric, 'mode': mode,
              'strategy': strategy}

    if save_output:
        assert rep_id
        save_sim_output(rep_id, output)
    if plot_traj or save_fig:
        if save_fig:
            assert rep_id
            plot_trajectories(output, rep_id=rep_id, save_fig=save_fig)
        else:
            plot_trajectories(output)
    if animate:
        assert rep_id
        animate_trajectories(rep_id, output)

    return output


def simulate_reactive(save_output=False, rep_id=None, plot_traj=False, save_fig=False, animate=False):

    mode = 'reactive'
    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s, bus_capacity,
                                     bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2, bus_cv_acc, pax_board_s,
                                     pax_alight_s, allow_early, slack_s, mode, bunch_threshold_s=bunch_threshold_s)

    bus_records, _ = simulator.simulate()
    metric = compute_metric(bus_records)

    allow_early_msg = 'Allow Earliness' if allow_early else 'No Earliness'
    slack_msg = 'Slack' if slack_s else 'No Slack'
    strategy = 'Reactive Insertion, {0}, {1}'.format(allow_early_msg, slack_msg)

    output = {'bus_records': bus_records, 'bus_addition_list': [], 'metric': metric, 'mode': mode,
              'strategy': strategy}

    if save_output:
        assert rep_id
        save_sim_output(rep_id, output)
    if plot_traj or save_fig:
        if save_fig:
            assert rep_id
            plot_trajectories(output, rep_id=rep_id, save_fig=save_fig)
        else:
            plot_trajectories(output)
    if animate:
        assert rep_id
        animate_trajectories(rep_id, output)

    return output


def simulate_preventive(save_output=False, rep_id=None, plot_traj=False, save_fig=False, animate=False,
                        bus_addition_list=None, num_reps=None, **kwargs):

    mode = 'preventive'

    if not bus_addition_list:
        # compute addition list if it is not given
        # get new insertion and update cumulative run time
        p_t = kwargs['p_t'] if 'p_t' in kwargs else p_threshold
        b_t = kwargs['b_t'] if 'b_t' in kwargs else bunch_threshold_s

        bus_addition_list = get_bus_addition_list(num_reps, p_t=p_t, b_t=b_t)

    # run a final simulation with bus_addition_list
    simulator = TransitLineSimulator(max_clk_s, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                     bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                     bus_cv_acc, pax_board_s, pax_alight_s, allow_early, slack_s, mode,
                                     bus_addition_list=bus_addition_list, delay_start_s=0)

    bus_records, _ = simulator.simulate()

    metric = compute_metric(bus_records)

    allow_early_msg = 'Allow Earliness' if allow_early else 'No Earliness'
    slack_msg = 'Slack' if slack_s else 'No Slack'
    strategy = 'Preventive Insertion, {0}, {1}'.format(allow_early_msg, slack_msg)

    output = {'bus_records': bus_records, 'bus_addition_list': bus_addition_list, 'metric': metric, 'mode': mode,
              'strategy': strategy}

    if save_output:
        assert rep_id
        save_sim_output(rep_id, output)
    if plot_traj or save_fig:
        if save_fig:
            assert rep_id
            plot_trajectories(output, rep_id=rep_id, save_fig=save_fig)
        else:
            plot_trajectories(output)
    if animate:
        assert rep_id
        animate_trajectories(rep_id, output)

    return output


def get_bus_addition_list(num_reps, **kwargs):
    """
    this function obtains a bus addition list based
    :param num_reps: number of replications
    :param kwargs: {p_t, b_t}
    :return:
    """

    mode = 'preventive'

    # initialize strategy
    delay_start_s = 0
    cum_time = 0
    bus_addition_list = []
    p_list = []
    inserted_bus_ids = set()

    # while there is a complete time window (run_period) for scheduling insertion
    while max_clk_s - cum_time > run_period:
        # print('Cumulative time: {0}'.format(cum_time))
        # dictionary to hold delays of all replications
        cum_delays = {}

        # max simulation time for this batch
        sim_time = cum_time + run_period

        # run simulations and store output
        for _ in range(num_reps):
            simulator = TransitLineSimulator(sim_time, num_stops, pax_hr, stop_spacing_m, num_buses, headway_s,
                                             bus_capacity, bus_mean_speed_kmh, bus_cv_speed, bus_mean_acc_ms2,
                                             bus_cv_acc, pax_board_s, pax_alight_s, allow_early, slack_s, mode,
                                             bus_addition_list=bus_addition_list, delay_start_s=delay_start_s)
            bus_records, delays = simulator.simulate()
            del simulator

            for key in delays:
                cum_delays.setdefault(key, []).append(delays[key])

        # get new insertion and update cumulative run time
        p_t = kwargs['p_t'] if 'p_t' in kwargs else p_threshold
        b_t = kwargs['b_t'] if 'b_t' in kwargs else bunch_threshold_s

        candidate = predict_bunching(cum_delays, inserted_bus_ids, p_t, b_t)

        if candidate:
            p = candidate[0]
            new_addition = candidate[1]
            bus_addition_list.append(new_addition)
            p_list.append(p)
            cum_time = new_addition[0]
        else:
            cum_time += run_period

        # delay computation for next batch starts at current cum_time
        delay_start_s = cum_time

    return bus_addition_list


def param_sweep_preventive(p_thresholds, bunch_thresholds, num_reps, save_sweep=True, sweep_id=1):
    """
    sweeps over parameters p_threshold and bunch_threshold_s and stores metric
    :param p_thresholds: list of p_threshold values with 0 < p < 1
    :param bunch_thresholds: list of bunch_threshold_s values with bunch_threshold_s <= headway
    :param num_reps: number of replications
    :param save_sweep: save avg_metrics in pickle
    :param sweep_id: pickle id
    :return avg_metrics: {p_t, b_t}: avg_metric}
    """

    # key: (p_t, b_t), value: metric
    avg_metrics = {}
    # sweep
    for p_t in p_thresholds:
        for b_t in bunch_thresholds:
            print('Processing p_t: {0}, b_t: {1}'.format(p_t, b_t))
            avg_metric = compute_avg_metric(simulate_preventive, num_reps, p_t=p_t, b_t=b_t)
            avg_metrics[(p_t, b_t)] = avg_metric

    if save_sweep:
        filename = '{0}/../avg_metrics_{1}.cpkl'.format(path, sweep_id)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(avg_metrics, f)

    return avg_metrics


def compute_metric(bus_records):
    """
    computes the metric (mean_speed, num_additions)
    :param bus_records: {bus_id: (clk, abs_dist, schedule_t, schedule_dist), ...}
    :return metric: (mean_speed, num_additions)
    """

    # store cumulative sum of x and t to apply definitions
    sum_x = 0
    sum_t = 0

    # get number of insertions
    bus_status_list = []

    # iterate over every bus trajectory
    for bus_id in bus_records:

        bus_record = list(zip(*bus_records[bus_id]))
        clk = bus_record[0]

        bus_status_list.append(tuple([bus_id, 'addition', clk[0]]))
        bus_status_list.append(tuple([bus_id, 'removal', clk[-1]]))

        abs_dist = bus_record[1]
        delta_x = abs_dist[-1] - abs_dist[0]
        delta_t = clk[-1] - clk[0]

        sum_x += delta_x
        sum_t += delta_t

    # convert units to km/h
    mean_speed = (sum_x / sum_t) * 3.6

    bus_status_list.sort(key=lambda x: x[2])
    num_active = 0
    max_active = 0
    for bus_status in bus_status_list:
        if bus_status[1] == 'addition':
            num_active += 1
        if bus_status[1] == 'removal':
            num_active -= 1
        if num_active > max_active:
            max_active = num_active

    metric = tuple([mean_speed, max_active])

    return metric


def compute_avg_metric(function, num_reps, p_t=None, b_t=None):
    """
    :param function: choose from simulate_no_insertion, simulate_reactive or simulate_preventive
    :param num_reps: number of replications
    :param p_t: probability threshold for preventive (used in parameter sweep)
    :param b_t: bunching threshold for preventive (used in parameter sweep)
    :return:
    """

    metrics = []

    assert (function == simulate_no_insertion or function == simulate_reactive or function == simulate_preventive)
    if function == simulate_preventive:
        bus_addition_list = get_bus_addition_list(num_reps, p_t=p_t, b_t=b_t)

    for _ in range(num_reps):
        if function == simulate_preventive:
            output = function(p_t=p_t, b_t=b_t, bus_addition_list=bus_addition_list)
        else:
            output = function()

        metric = output['metric']
        metrics.append(metric)

    avg_metric = [sum(i) / len(i) for i in zip(*metrics)]

    return avg_metric


def predict_bunching(cum_delays, inserted_bus_ids, p_t, b_t):
    """
    computes the insertion location with maximum probability if p_threshold is satisfied
    :param cum_delays: dict of rep delays with key (schedule_t, schedule_dist, schedule_stop_id, retired_bus_id)
    :param inserted_bus_ids: set of inserted bus ids (cannot insert on same bus more than once)
    :param p_t: probability threshold with 0 < pt < 1
    :param b_t: bunching threshold with b_t <= headway
    :return candidate: (p, key from cum_delays)
    """

    # iterate over all insertion candidates and append to (p, key) to candidate list
    # is p_threshold is satisfied
    candidates = []
    for key in cum_delays:
        delay_list = cum_delays[key]
        p = sum(i > b_t for i in delay_list) / len(delay_list)
        if p >= p_t:
            candidates.append(tuple([p, key]))

    # sort by p
    candidates.sort(key=lambda x: x[0])

    # return candidate with largest p that is not in inserted_bus_ids
    for candidate in candidates:
        if candidate[1][-1] not in inserted_bus_ids:
            inserted_bus_ids.add(candidate[1][-1])
            return candidate

    return None


def save_sim_output(rep_id, output):

    filename = '{0}/../output/rep_{1}.cpkl'.format(path, rep_id)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(output, file)


def load_sim_output(rep_id):

    filename = '{0}/../output/rep_{1}.cpkl'.format(path, rep_id)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'rb') as file:
        output = pickle.load(file)

    return output


def plot_trajectories(output, rep_id=None, save_fig=False):
    """
    plots bus trajectories
    :param output: output dictionary
    :param rep_id: replication id
    :param save_fig: True if save fig, False otherwise
    :return:
    """

    # unpack output
    bus_records = output['bus_records']
    metric = output['metric']
    strategy = output['strategy']

    # used for labeling traj and sched only once
    labeled = False
    # plot each trajectory
    for bus_id in bus_records:
        bus_record = list(zip(*bus_records[bus_id]))
        try:
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
        except IndexError:
            pass

    plt.legend(loc=2)
    plt.title('Strategy: {0} \n Mean Commercial Speed: {1:.2f} km/h, No. of Buses Needed: {2}'.format(strategy,
                                                                                                      metric[0],
                                                                                                      metric[1]))
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Distance [m]', fontsize=16)
    plt.ylim([0, 100000])

    if save_fig:
        filename = '{0}/../figures/rep_{1}.png'.format(path, rep_id)
        plt.savefig(filename, dpi=600)
        plt.close("all")
        return
    plt.show()


def animate_trajectories(rep_id, output):
    """
    plots bus trajectories
    :param rep_id: replication id
    :param output: output dictionary
    :return:
    """

    bus_records = output['bus_records']
    bus_addition_list = output['bus_addition_list']
    metric = output['metric']
    strategy = output['strategy']

    # set track
    r = 1
    angles = np.linspace(0, 2 * np.pi, num_stops)
    locs = [[r * np.cos(angle), r * np.sin(angle)] for angle in angles]
    x, y = zip(*locs)

    # save stop location by stop id
    loc_dict = {}
    for idx, loc in enumerate(locs):
        loc_dict[idx] = loc

    # draw path
    fig, ax = plt.subplots(figsize=(8, 8))
    route = plt.Circle((0, 0), 1, color='k', fill=False)

    # build frames
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

    # to find when to label a bus as retired
    if bus_addition_list:
        for addition in bus_addition_list:
            frames.append([addition[3], addition[0], -2])
    frames.sort(key=lambda step: step[1])

    os.makedirs('{0}/../animations/rep_{1}/'.format(path, rep_id), exist_ok=True)

    # save each frame
    cur_locs = {}
    bus_colors = {}
    last_frame_clk = 0
    frame_ct = 1
    for idx, frame in enumerate(frames):
        if frame[-1] == -2:
            bus_colors[frame[0]] = 'r'
            continue

        cur_locs[frame[0]] = frame[2]

        if frame[0] not in bus_colors:
            bus_colors[frame[0]] = 'b'

        clk = frame[1]
        ax.cla()

        for bus_id in cur_locs:
            px, py = loc_dict[cur_locs[bus_id]]
            ax.scatter(px, py, marker='s', color=bus_colors[bus_id], s=100)

        ax.scatter(x, y, color='k', marker='o')
        ax.add_artist(route)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.scatter([-0.12], [-0.265], marker='s', s=100, color='b')
        ax.annotate('In Service', xy=(-0.04, -0.3))

        ax.scatter([-0.12], [-0.365], marker='s', s=100, color='r')
        ax.annotate('Out of Service', xy=(-0.04, -0.4))
        ax.set_title('Strategy: {0}'.format(strategy))
        ax.text(0, 0, 'Mean Commercial Speed: {0:.2f} km/h \n '
                      'Number of Buses Running: {1} \n Clock Time: {2:.2f} s'.format(metric[0], len(cur_locs), clk),
                fontsize=15, horizontalalignment='center')

        for _ in range(int(clk) - last_frame_clk):
            fig.savefig('{0}/../animations/rep_{1}/{2}.png'.format(path, rep_id, frame_ct), dpi=175)
            frame_ct += 1
        last_frame_clk = int(clk)

        if frame[-1] == -1:
            del cur_locs[frame[0]]


if __name__ == '__main__':
    pass
