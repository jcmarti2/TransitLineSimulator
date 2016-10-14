import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

__author__ = 'juan carlos martinez mori'

# set output directory
config_dir = '{0}/../config/'.format(os.path.dirname(os.path.realpath(__file__)))
output_dir = '{0}/../output/'.format(os.path.dirname(os.path.realpath(__file__)))


def build_trajectories(rep_id):
    if os.path.isfile('{0}rep{1}_stops.txt'.format(config_dir, rep_id)):
        stop_dist = {}
        with open('{0}rep{1}_stops.txt'.format(config_dir, rep_id), 'r+') as file:
            for row in file:
                row = row.strip('\n').split(';')
                # input(row)
                stop_dist[row[0]] = float(row[1])
    else:
        raise Exception('There is no stops configuration file for rep{0} ...'.format(rep_id))

    if os.path.isfile('{0}rep{1}_output.txt'.format(output_dir, rep_id)):
        trajs = {}
        with open('{0}rep{1}_output.txt'.format(output_dir, rep_id), 'r+') as file:
            for row in file:
                row = row.strip('\n').split(',')
                if row[2] == 'bus_arr' or row[2] == 'bus_dept':
                    if row[5] not in trajs:
                        trajs[row[5]] = []
                    # input(row[3])
                    trajs[row[5]].append([row[1], stop_dist[row[3]]])

        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for bus in trajs:
            tss = []
            dists = []
            for stop in trajs[bus]:
                tss.append(float(stop[0]))
                dists.append(stop[1])
            ax.plot(tss, dists, 'k')
        ax.set_title('Transit Line Vehicle Trajectories')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Distance [m] ')
        plt.show()


    else:
        raise Exception('There is no bus output file for rep{0} ...'.format(rep_id))


def load_trajectories(rep_id):
    pass

def plot_trajectories(rep_id):
    pass

def plot_bunching_time_loc():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    with open('/Users/jotaceemeeme/Desktop/GitHub/TransitLineSimulator/output/rep_output_temp.txt', 'r+') as file:

        for line in file:
            line = line.strip('\n').split(',')
            ax.scatter(float(line[1]), float(line[0]))
    ax.set_title('Bus Addition Triggering Scatter Plot')
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Distance [m] ')
    plt.show()


plot_bunching_time_loc()
# build_trajectories(1)