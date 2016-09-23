# TransitLineSimulator

## about this repository
this repository contains the TransitLineSimulator

## directories
### /config: this directory contains the simulation configuration files
  in the following descriptions '{0}' denotes the (unique) replication id
  - rep{0}_rep.txt: [max_clock_time, headway, pax_board_t, pax_alight_t, bunch_threshold, bus_addition_stops_ahead]
    * max_clock_time: maximum simulation clock time [s]
    * headway: time headway between bus departures [s]
    * pax_board_t: boarding time per passenger [s]
    * pax_alight_t: alighting time per passenger [s]
    * bunch_treshold: time delay for bus addition triggering [s]
    * bus_addition_stops_ahead: int number of stops ahead for bus addition []
  - rep{0}_stops.txt: [bus_id, bus_capacity, mean_cruise_speed, cv_cruise_speed, acc_rate, cv_acc_rate, stop_list, stop_slack]
    * bus_id: unique bus run id []
    * bus_capacity: int bus passenger capacity []
    * mean_cruise_speed: mean bus cruise speed [km/h]
    * cv_cruise_speed: coefficient of variation for cruise speed [] (if no variation, set to small number but not 0)
    * acc_rate: mean bus accelerationrate [m/s^2]
    * cv_acc_rate: coefficient of variation for acceleration rate [] (if no variation, set to small number but not 0)
    * stop_list:
    * stop_slack:
  - rep{0}_buses.txt

[bus_id, bus_capacity, mean_cruise_speed, cv_cruise_speed, acc_rate
                                                cv_acc_rate, stop_list, stop_slack]

### /output: this directory contains the simulation output files

### /src: this directory contains the simulation source code
  - TransitLine.py
  - simulate_transit_line.py
