'''
A command line script to allow running instances of test11() from vacuum.py
'''

import argparse
from vacuum import test11

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sensor_radius_min", type=int, help="Minimum drone sensor radius that should be considered")
    parser.add_argument("sensor_radius_max", type=int, help="Maximum drone sensor radius that should be considered")
    args = parser.parse_args()

    test11(args.sensor_radius_min, args.sensor_radius_max, showPlot=False)

if __name__ == "__main__":
    main()
