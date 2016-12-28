__author__ = 'juan carlos martinez mori'

# ================================================================== #
# SIMULATION PARAMETERS
#   the default value for the simulation run parameters is set next
#   alternative values may be passed in kwargs
# ================================================================== #
max_clk_s = 14400           # maximum simulation time [s]
allow_early = False         # True if allowing for earliness, False otherwise
slack_s = 0                 # slack per stop [s]
p_threshold = 0.9           # probability threshold for bus addition
bunch_threshold_s = 350     # bus bunching threshold [s] (default in simulator is False)
num_reps = 100              # number of replications
window_s = 900                # time window for heuristic [s]

# ================================================================== #
# MODEL PARAMETERS
#   these fixed value for the model parameters is set next
# ================================================================== #
num_stops = 24              # number of stops []
pax_hr = 50                 # pax per hour at each stop [pax/hr]
stop_spacing_m = 1000       # spacing between stops [m]
num_buses = 8               # number of buses []
headway_s = 432.5           # headway between buses [s]
bus_capacity = 100          # bus capacity [pax]
bus_mean_speed_kmh = 30     # mean bus cruise speed [km/h]
bus_cv_speed = 0.1          # coefficient of variation for bus speed
bus_mean_acc_ms2 = 1        # mean bus acceleration [m/s2]
bus_cv_acc = 0.1            # coefficient of variation for bus acceleration
pax_board_s = 4             # boarding time per pax [s/pax]
pax_alight_s = 4            # alighting time per pax [s/pax]

