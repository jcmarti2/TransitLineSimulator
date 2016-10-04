import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

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
                input(row)
                stop_dist[row[0]] = float(row[1])
    else:
        raise Exception('There is no stops configuration file for rep{0} ...'.format(rep_id))

    if os.path.isfile('{0}rep{1}_output.txt'.format(output_dir, rep_id)):
        trajs = {}
        with open('{0}rep{1}_output.txt'.format(output_dir, rep_id), 'r+') as file:
            for row in file:
                row = row.strip('\n').split(',')
                if row[2] == 'bus_arr' or row[2] == 'bus_dept':
                    if row[3] not in trajs:
                        trajs[row[3]] = []
                    input(row[3])
                    trajs[row[3]].append([row[1], stop_dist[row[3]]])

        plt.figure()
        for bus in trajs:
            tss = []
            dists = []
            for stop in trajs[bus]:
                tss.append(float(stop[0]))
                dists.append(stop[1])
            plt.plot(tss, dists)
        plt.show()


    else:
        raise Exception('There is no bus output file for rep{0} ...'.format(rep_id))


def load_trajectories(rep_id):
    pass

def plot_trajectories():
    pass

build_trajectories(1)