import os

__author__ = 'juan carlos martinez mori'

# ------------------------------ #
#           USER INPUT           #
# ------------------------------ #

rep_id = 1                      # replication id

# -- replication -- #
replication = True
max_clk = 36000                       # maximum simulation clock time; float or int [s]
headway = 600                        # time headway between bus departures; float or int [s]
pax_board_t = 2                      # boarding time per passenger; float or int [s]
pax_alight_t = 2                     # alighting time per passenger; float or int [s]
bunch_threshold = 300                # time delay for bus addition triggering; float or int [s]
bus_addition_stops_ahead = 3         # number of stops ahead for bus addition; int []


# ----- buses ----- #
buses = True
num_buses = 100

# uniform bus configs (simple)
buses_uniform = True                 # True if bus building is uniform, False otherwise; bool
bus_capacity = 30                    # passenger capacity of bus; int [pax]
mean_cruise_speed = 50               # mean cruise speed; float or int [km/h]
cv_cruise_speed = 2                  # coefficient of variation of cruise speed; float or int [] (small if no r.v.)
mean_acc_rate = 4                    # mean acceleration rate; float or int [m/s^2]
cv_acc_rate = 0.5                    # coefficient of variation of acceleration rate; float or int [] (small if no r.v.)
stop_list = list(range(12))          # stops list; [stop id of stop for stop in stops]
stop_slack = [0 for _ in range(12)]  # stops slack list [slack at stop for stop in stops]; int or float [s]

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
stops = False
# uniform stop configs (simple)
stops_uniform = True
# custom stop configs (same types as above, specify by hand).
# len of all lists must be the same.
stops_custom = False

# ------------------------------ #
#         NO USER INPUT          #
# ------------------------------ #

config_dir = '{0}/../config/'.format(os.path.dirname(os.path.realpath(__file__)))

if replication:
    with open('{0}rep{1}_rep.txt'.format(config_dir, rep_id), 'w+') as file:
        file.write('{0};{1};{2};{3};{4};{5}\n'.format(max_clk, headway, pax_board_t, pax_alight_t, bunch_threshold,
                                                      bus_addition_stops_ahead))

if buses:
    if (buses_uniform and buses_custom) or (not buses_uniform and not buses_custom):
        raise Exception('Specify only one of uniform or customized bus building ...')
    with open('{0}rep{1}_buses.txt'.format(config_dir, rep_id), 'w+') as file:
        if buses_uniform:
            for bus_id in range(num_buses):
                file.write('{0};{1};{2};{3};{4};{5};{6};{7}\n'.format(bus_id, bus_capacity, mean_cruise_speed,
                                                                      cv_cruise_speed, mean_acc_rate, cv_acc_rate,
                                                                      stop_list, stop_slack))
        elif buses_custom:
            for bus_id in range(num_buses):
                file.write('{0};{1};{2};{3};{4};{5};{6};{7}\n'.format(bus_id, bus_capacities[bus_id],
                                                                      mean_cruise_speeds[bus_id],
                                                                      cv_cruise_speeds[bus_id], mean_acc_rates[bus_id],
                                                                      cv_acc_rates[bus_id], stop_lists[bus_id],
                                                                      stop_slacks[bus_id]))

if stops:
    if (stops_uniform and stops_custom) or (not stops_uniform and not stops_custom):
        raise Exception('Specify only one of uniform or customized stop building ...')
    with open('{0}rep{1}_stops.txt'.format(config_dir, rep_id), 'w+') as file:
        pass
