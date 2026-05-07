import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import json
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VERONA"))
from ada_verona.analysis.report_creator import ReportCreator

def str_int_tuple(t):
    try:
        name, value = t.split(',')

        seed = None if value.lower() == 'none' else int(value)
        return (str(name), seed)
    except ValueError:
        raise argparse.ArgumentTypeError("Format must be name,value (e.g., 'pixelreg,0' or 'pixelreg,None')")

models = json.load(open("models.json", "r"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stats and plots for RDs")

    parser.add_argument("--models", nargs='+', type=str_int_tuple)
    args = parser.parse_args()

    TODO: "make a separate json with the experiment paths. use that"
    print(args.models)