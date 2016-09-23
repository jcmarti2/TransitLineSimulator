# TransitLineSimulator
## author
juan carlos martinez mori

## about this repository
this repository contains the TransitLineSimulator. to use this package, please refer to the sections

## directories
### /config: this directory contains the simulation configuration files
in the following descriptions '{0}' denotes the (unique) replication id. the columns of each file are specified.
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
  - rep{0}_buses.txt: [stop_id, abs_distance, board_demand, [subseq_alight_demand]]
    * stop_id: unique stop id []
    * abs_distance: absolute stop distance along the bus line from the origin [m]
    * board_demand: passenger demand [pax/hr]
    * [subseq_alight_demand]: list containing alight demand from this stop to the subsequent stops [pax/hr]

### /output: this directory contains the simulation output files
in the following descriptions '{0}' denotes the (unique) replication id. the columns of each file are specified. there are four event types: {'0': initialization, '1': bus arrival at stop, '2': bus departure from stop, '3': passenger arrival}. the format for each event type in each output entry is separated by 0|1|2|3.
  - rep{0}_output.txt: [ent, clk, event_type, stop_id, stop_pax, bus_id, bus_type, bus_pax]
    * ent: output entry counter []. ent|ent|ent|ent
    * clk: clock time [s]. clk|clk|clk|clk
    * event_type: []. 0|1|2|3
    * stop_id: stop id []. ''|stop_id|stop_id|stop_id
    * stop_pax: number of passengers at stop []. ''|stop_pax|stop_pax|stop_pax
    * bus_id: bus id []. ''|bus_id|bus_id|''
    * bus_type: bus type []. ''|bus_type|bus_type|''
    * bus_pax: number of passengers in bus. ''|bus_pax|bus_pax|''

### /src: this directory contains the simulation source code
  - TransitLine.py: contains simulator
  - simulate_transit_line.py: runs simulator

## disclaimer
this package is freely available, but please cite when appropriate. note that this package is provided 'as-is'. while i have put significant effort verifying this package, i cannot account for all possible edge cases at this moment.
