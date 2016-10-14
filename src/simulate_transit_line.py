import os
from TransitLine import TransitLine

__author__ = 'juan carlos martinez mori'

# ------------------------------ #
#           USER INPUT           #
# ------------------------------ #

rep_ids = range(10000)
overwrite = True

# ------------------------------ #
#         NO USER INPUT          #
# ------------------------------ #


tracker_file = '{0}/../rep_tracker.txt'.format(os.path.dirname(os.path.realpath(__file__)))

for rep_id in rep_ids:
    with open(tracker_file, 'r+') as tracker:
        run = False
        for line in tracker:
            if line == 'rep{0}'.format(rep_id):
                run = True
                break
    if not run or (run and overwrite):
        simulator = TransitLine()
        simulator.simulate(rep_id)
    with open(tracker_file, 'a+') as tracker:
        if not run:
            tracker.write('rep{0}\n'.format(rep_id))


