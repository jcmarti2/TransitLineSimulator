__author__ = 'juan carlos martinez mori'

max_clk_s = 14400           # maximum simulation time [s]
run_period = 900            # period for running bus addition strategy [s]
bunch_threshold_s = 312     # bus bunching threshold [s] (default in simulator is False)
p_threshold = 0.5           # probability threshold for bus addition
allow_early = False         # True if allowing for earliness, False otherwise
slack_s = 0                 # slack per stop [s]

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


# temporary parameters
addition_list = [(3248.19, 19000.0, 20, 0), (5162.64, 29000.0, 6, 1), (6187.92, 33000.0, 10, 2),
                 (7361.39, 38000.0, 15, 3), (7485.42, 33000.0, 10, 5), (7497.5, 36000.0, 13, 4),
                 (8646.81, 35000.0, 12, 7), (9251.67, 42000.0, 19, 6), (10213.33, 66000.0, 20, 8),
                 (10497.64, 65000.0, 19, 9), (11671.11, 70000.0, 0, 10), (11795.14, 65000.0, 19, 13),
                 (12696.39, 74000.0, 4, 11), (12944.45, 64000.0, 18, 14), (13413.2, 73000.0, 3, 12)]
