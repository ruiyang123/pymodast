"""
    In order to use this script, we need to define :

        path_to_Etude
        scenario_name
        targets : a two dimensional liste [[Sections],[cals],varnames]
        path_to_PXs : str
            the path of the folder for storing all DOEs
        output_path : str
            the path of output folder
        out_file_name
        all_name : there exist two mode.  mode 0 : outputs defined by the targets. mode1 : all sections are outputs

    At last, we can generate lots of datasets with format .csv for the meta-modeling stored in the output_path
"""
from crue10.etude import Etude
from crue10.utils.multiple_runs import launch_scenario_modifications
from copy import deepcopy
import numpy as np
import pandas as pd
import os
import utils
from params_run_simulation_ import *

def generate_modifications(path_to_PX, etude, scenario):
    """ Generate modifications listes for simulations

    :param path_to_PX: str
        path to the experimental design
    :param etude : etude object
        a study of FudaaCrue (.etu)
    :param scenario : scenario object
        a scenario of <etude>
    :return: dict
        a dict contains the all the modifications
    """
    samples = pd.read_csv(path_to_PX,index_col=0)
    print(samples)
    samples_array = samples.to_numpy()
    sample_size = len(samples)
    variable_names = samples.columns
    modifications_liste = []

    for i in range(sample_size):
        modifications = {'run_id': 'Iter%i' % i}
        for loi_frottement in scenario.modele.get_liste_lois_frottement(ignore_sto=True):
            fk_id = loi_frottement.id
            loi_frottement = scenario.modele.get_loi_frottement(fk_id)
            new_strickler = loi_frottement.loi_Fk[0, 1]
            index = 0
            for variable_name in variable_names:
                if fk_id == variable_name:
                    new_strickler = samples_array[i][index]
                index += 1
            modifications[fk_id] = new_strickler
        modifications_liste.append(modifications)
    return modifications_liste



etude = Etude(path_to_Etude)
etude.read_all()

scenario = etude.get_scenario(scenario_name)
scenario.remove_all_runs(sleep=1.0)


def get_all_sections_names(etude):
    modele = etude.get_modele('Mo_CE2016_PR1_Juin_2016')
    branches = modele.get_branches_liste_entre_noeuds('Nd_RET151.520', 'Nd_CAA145.950PR1')
    section_names = []
    for branche in branches:
        for section in branche.liste_sections_dans_branche:
            section_names.append(section.id)
    return section_names



def apply_modifications(modifications):
    return scenario.get_function_apply_modifications(etude)(modifications)

if __name__ == "__main__" :
    all_name = 0
    if os.path.exists(output_path) == 0:
        os.mkdir(output_path)

    # create a list consists of all file names of the file path.
    for i, j, k in os.walk(path_to_PXs):
        file_names = k

    # if you only want to run simulation for one single dataset you can define
    # file_names = [file_name]
    for file_name in file_names:
        print(file_name)
        etude = Etude(path_to_Etude)
        etude.read_all()

        scenario = etude.get_scenario(scenario_name)
        scenario.remove_all_runs(sleep=1.0)
        path_to_PX = os.path.join(path_to_PXs,file_name)

        # get methode name
        for i in path_to_PX.split('_'):
            if "methode" in i:
                methode_name = i.split(".")[0]
            else:
                methode_name = "opt"
        # get PX number
        for i in file_name.split('_'):
            if "PX" in i:
                if i == "PX":
                    px_number = 0
                else :
                    px_number = int(i.split('X')[-1])

        if all_name == 0:
            #out_file_name = "dataset{}_size{}_{}.csv".format(px_number,sample_size, methode_name)
            out_file_name = file_name
        else:
            out_file_name = "allSect_dataset_{}.csv".format(methode_name)
        for a, b, c in os.walk(output_path):
            out_filenames = c

        if out_file_name not in out_filenames :
            if all_name == 0 :
                targets = utils.get_targets_from_aoc(file_aoc,file_lhpt)
            else :
                targets = [get_all_sections_names(etude), ['Cc_P009_LE20160426', 'Cc_P012_LE20160202'], 'Z']


            samples = pd.read_csv(path_to_PX, index_col=0)
            modifications = generate_modifications(path_to_PX, etude, scenario)
            runs_liste = launch_scenario_modifications(apply_modifications, modifications)

            #for run_id, run in enumerate(runs_liste):
            #    if run.nb_erreurs_calcul() != 0:
            #        del runs_liste[run_id]


            for run in runs_liste:
                scenario.add_run(run)
            scenario.set_current_run_id(runs_liste[-1].id)
            etude.write_etu()

            sample_size = len(modifications)
            output_length = len(targets[0])

            for cal in targets[1]:
                Z = np.zeros((sample_size, output_length))
                for index, run in enumerate(runs_liste):
                    if run.nb_erreurs_calcul()==0:
                        results = run.get_results()
                        res = results.get_res_steady(cal)
                        emh_types = []
                        for emh_name in targets[0]:
                            emh_types.append(results.emh_type(emh_name))
                        values = []
                        for i,(emh_name,emh_type) in enumerate(zip(targets[0],emh_types)):
                            emh_pos = results.get_emh_position(emh_type, emh_name)
                            var_pos = results.get_variable_position(emh_type, targets[2])
                            values.append(res[emh_type][emh_pos, var_pos])


                        Z[index] = np.array(values)

                for i in range(output_length):
                    target_name = cal+'/'+targets[0][i]
                    samples[target_name] = Z[:,i]

            samples.to_csv(os.path.join(output_path,out_file_name),index=False)



