import os

__author__ = 'juan carlos martinez mori'

# ------------------------------ #
#           USER INPUT           #
# ------------------------------ #

rep_ids = list(range(10000))            # replication id

# -- replication -- #
replication = True
max_clk = 36000                      # maximum simulation clock time; float or int [s]
headway = 300                        # time headway between bus departures; float or int [s]
pax_board_t = 4                      # boarding time per passenger; float or int [s]
pax_alight_t = 4                     # alighting time per passenger; float or int [s]
bunch_threshold = 50                # time delay for bus addition triggering; float or int [s]
bus_addition_stops_ahead = 2         # number of stops ahead for bus addition; int []


# ----- buses ----- #
buses = True
num_buses = 12

# uniform bus configs (simple)
buses_uniform = True                 # True if bus building is uniform, False otherwise; bool
num_stops = 24                      # number of stops in line from origin to end; int []
bus_capacity = 200                   # passenger capacity of bus; int [pax]
mean_cruise_speed = 30               # mean cruise speed; float or int [km/h]
cv_cruise_speed = 2/30               # coefficient of variation of cruise speed; float or int [] (small if no r.v.)
mean_acc_rate = 1                    # mean acceleration rate; float or int [m/s^2]
cv_acc_rate = 0.1/1                  # coefficient of variation of acceleration rate; float or int [] (small if no r.v.)
stop_list = list(range(num_stops))   # stops list; [stop id of stop for stop in stops]
stop_slack = [0 for _ in stop_list]  # stops slack list [slack at stop for stop in stops]; int or float [s]

# custom bus configs (same types as above, specify by hand).
# len of all lists must be the same.
buses_custom = False
bus_capacities = []
mean_cruise_speeds = []
cv_cruise_speeds = []
mean_acc_rates = []
cv_acc_rates = []
stop_lists = []
stop_slacks = []


# ----- stops ----- #
stops = True
# uniform stop configs (simple)
stops_uniform = True                 # True if stop building is uniform, False otherwise; bool
num_stops = 24                       # number of stops in line from origin to end; int []
spacing = 1000                        # spacing between stops; float or int [m]
board_demand = 50                   # passenger demand; float or int [pax/hr]

# custom stop configs (same types as above, specify by hand).
# len of all lists must be the same.
stops_custom = False                 #
abs_distances = []                   #
board_demands = []                   #
subseq_alight_demands = []           # [stop_id + 1, stop_id + 2, ... ,end]; sum must be board_demand[stop_idx] [pax/hr]

# ------------------------------ #
#         NO USER INPUT          #
# ------------------------------ #

config_dir = '{0}/../config/'.format(os.path.dirname(os.path.realpath(__file__)))

for rep_id in rep_ids:

    if replication:
        with open('{0}rep{1}_rep.txt'.format(config_dir, rep_id), 'w+') as file:
            file.write('{0};{1};{2};{3};{4};{5}\n'.format(max_clk, headway, pax_board_t, pax_alight_t, bunch_threshold,
                                                          bus_addition_stops_ahead))

    if buses:
        if (buses_uniform and buses_custom) or (not buses_uniform and not buses_custom):
            raise Exception('Specify only one of uniform or customized bus building ...')
        with open('{0}rep{1}_buses.txt'.format(config_dir, rep_id), 'w+') as file:
            if buses_uniform:
                for bus_id in range(num_buses):  # starts at 0
                    bus_id = 'RR{0}'.format(bus_id)
                    file.write('{0};{1};{2};{3};{4};{5};{6};{7}\n'.format(bus_id, bus_capacity, mean_cruise_speed,
                                                                          cv_cruise_speed, mean_acc_rate, cv_acc_rate,
                                                                          stop_list, stop_slack))
            elif buses_custom:
                for bus_id in range(num_buses):  # starts at 0
                    bus_id = 'RR{0}'.format(bus_id)
                    file.write('{0};{1};{2};{3};{4};{5};{6};{7}\n'.format(bus_id, bus_capacities[bus_id],
                                                                          mean_cruise_speeds[bus_id],
                                                                          cv_cruise_speeds[bus_id],
                                                                          mean_acc_rates[bus_id],
                                                                          cv_acc_rates[bus_id], stop_lists[bus_id],
                                                                          stop_slacks[bus_id]))

    if stops:
        if (stops_uniform and stops_custom) or (not stops_uniform and not stops_custom):
            raise Exception('Specify only one of uniform or customized stop building ...')
        with open('{0}rep{1}_stops.txt'.format(config_dir, rep_id), 'w+') as file:
            if stops_uniform:
                for stop_id in range(num_stops):  # starts at 0
                    file.write('{0};{1};{2};{3}\n'.format(stop_id, spacing*stop_id, board_demand,
                                                          [board_demand/(num_stops - stop_id - 1)
                                                           for _ in range(num_stops - stop_id - 1)]))
            elif stops_custom:
                for stop_id in range(num_stops):  # starts at 0
                    file.write('{0};{1};{2};{3}\n'.format(stop_id, abs_distances[stop_id], board_demands[stop_id],
                                                          subseq_alight_demands[stop_id]))
