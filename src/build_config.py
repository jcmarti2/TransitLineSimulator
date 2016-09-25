import os

__author__ = 'juan carlos martinez mori'

# ------------------------------ #
#           USER INPUT           #
# ------------------------------ #

rep_id = 1                      # replication id

# replication
replication = True
max_clk = 3600                  # maximum simulation clock time; float or int [s]
headway = 600                   # time headway between bus departures; float or int [s]
pax_board_t = 2                 # boarding time per passenger; float or int [s]
pax_alight_t = 2                # alighting time per passenger; float or int [s]
bunch_threshold = 300           # time delay for bus addition triggering; float or int [s]
bus_addition_stops_ahead = 3    # number of stops ahead for bus addition; int []

# buses
buses = False

# stops
stops = False

# ------------------------------ #
#         NO USER INPUT          #
# ------------------------------ #

config_dir = '{0}/../config/'.format(os.path.dirname(os.path.realpath(__file__)))

if replication:
    with open('{0}rep{1}_rep.txt'.format(config_dir, rep_id), 'w+') as file:
        file.write('{0};{1};{2};{3};{4};{5}\n'.format(max_clk, headway, pax_board_t, pax_alight_t, bunch_threshold,
                                                      bus_addition_stops_ahead))

if buses:
    with open('{0}rep{1}_buses.txt'.format(config_dir, rep_id), 'w+') as file:
        pass

if stops:
    with open('{0}rep{1}_stops.txt'.format(config_dir, rep_id), 'w+') as file:
        pass
