import openturns as ot
import pandas as pd
import numpy as np
import os
from pymodast.utils import prepare_csv_for_2d, generate_px_2d, copy_file, get_inputs_from_aoc, get_inputs_from_csv, increase_domain


def generate_px(variables, ranges, output_path, size=None, method="LHS", px_name=None):
    """Generate a experimental design for a set of variables
    :param variables: list
        input variables for the meta-modeling
    :param ranges: dict
        the ranges (domains) for input variables, dict
    :param size: int
        the size of the experimental design, int
    :param method: str
        the method to create the experimental design ("LHS","HaltonSequence"),
    :param output_path: str
        the path for saving the design
    :return: None
        but the experimental design will be saved as .csv file in the output_path
    """
    input_length = len(variables)
    if not size:
        size = 10 * input_length

    if method == "LHS":
        dist = ot.ComposedDistribution([ot.Uniform(0, 1)] * input_length)
        samples = ot.LHSExperiment(dist, size).generate()
    elif method == "Halton":
        sequence = ot.HaltonSequence(input_length)
        samples = sequence.generate(size + 1)
    elif method == "Sobol":
        sequence = ot.SobolSequence(input_length)
        samples = sequence.generate(size + 1)
    elif method == "C2":
        dist = ot.ComposedDistribution([ot.Uniform(0, 1)] * input_length)
        lhs = ot.LHSExperiment(dist, size)
        lhs.setAlwaysShuffle(True)
        spacefilling = ot.SpaceFillingC2()
        n = 500
        optimalLHSAlgorithm = ot.MonteCarloLHS(lhs, n, spacefilling)
        samples = optimalLHSAlgorithm.generate()
    elif method == "MinDist":
        dist = ot.ComposedDistribution([ot.Uniform(0, 1)] * input_length)
        lhs = ot.LHSExperiment(dist, size)
        lhs.setAlwaysShuffle(True)
        spacefilling = ot.SpaceFillingMinDist()
        n = 500
        optimalLHSAlgorithm = ot.MonteCarloLHS(lhs, n, spacefilling)
        samples = optimalLHSAlgorithm.generate()
    elif method == "PhiP":
        dist = ot.ComposedDistribution([ot.Uniform(0, 1)] * input_length)
        lhs = ot.LHSExperiment(dist, size)
        lhs.setAlwaysShuffle(True)  # randomized
        # Defining space fillings
        spacefilling = ot.SpaceFillingPhiP(50)
        # RandomBruteForce MonteCarlo with N designs (LHS with C2 optimization)
        n = 500
        optimalLHSAlgorithm = ot.MonteCarloLHS(lhs, n, spacefilling)
        samples = optimalLHSAlgorithm.generate()
    for index, key in enumerate(ranges):
        lower = ranges[key][0]
        upper = ranges[key][1]
        samples[:, index] = samples[:, index] * (upper - lower) + lower
    if os.path.exists(output_path) == 0:
        os.mkdir(output_path)
    if not px_name:
        px_name = "PX_size{}_method{}.csv".format(size, method)
    samples.exportToCSVFile(os.path.join(output_path, px_name))
    samples = pd.read_csv(os.path.join(output_path, px_name), delimiter=";")
    samples.columns = variables
    samples.to_csv(os.path.join(output_path, px_name))
    return None


def prepare_cases_2d(px_path, i2s_path, cas_path, output_path, iter_name="iter"):
    prepare_csv_for_2d(px_path, px_path, name=iter_name)
    generate_px_2d(i2s_path, px_path, output_path, sep=",", encoding="utf-8")
    cas_folder = os.path.dirname(cas_path)
    cas_name = os.path.basename(cas_path)
    copy_file(cas_folder, output_path, cas_name)
    return None


if __name__ == "__main__":
    # 1D
    # get inputs names and domains
    aoc_path = "./Etu_CE2016_Conc/Etu_CE2016.aoc.xml"
    variables, ranges = get_inputs_from_aoc(aoc_path)
    ranges = increase_domain(ranges)
    size = 10 * len(variables)
    method = 'MinDist'
    output_path = "./CE/PX/"  # store output(DOE) format .csv
    generate_px(variables, ranges, output_path, size=size, method=method)

    # 2D
    # get inputs
    file_path = "./T2D_BV2016_6LE_19Fk/zones_frottement.csv"
    delimiter = ";"
    variables, ranges = get_inputs_from_csv(file_path, delimiter=delimiter)
    # generate a DOE
    size = 190
    method = "MinDist"
    output_path = "./BV/PX/"
    generate_px(variables, ranges, output_path, size=size, method=method)
    px_name = "PX_size{}_method{}.csv".format(size, method)
    px_path = os.path.join(output_path, px_name)
    i2s_path = './T2D_BV2016_6LE_19Fk/zones_frottements_BV2016.i2s'
    cas_path = "./T2D_BV2016_6LE_19Fk/t2d.cas"
    prepare_cases_2d(px_path, i2s_path, cas_path, output_path)
