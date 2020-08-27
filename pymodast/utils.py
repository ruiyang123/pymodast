from lxml import etree
import numpy as np
import pandas as pd
import os.path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO


def get_input_length(dataframe):
    for i, name in enumerate(dataframe.columns):
        if "/" in name:
            return i
    return None


def get_num_crue(dataframe):
    crue_names = []
    for name in dataframe.columns:
        if "/" in name:
            crue_names.append(name.split("/")[0])
    return len(set(crue_names))


def get_inputs_from_csv(file_path, delimiter=";"):
    fk = pd.read_csv(file_path, delimiter=delimiter)
    variables = []
    ranges = {}
    for index, row in fk.iterrows():
        variables.append(row["nom_zone"])
        inf = row["min"]
        sup = row["max"]
        ranges[row["nom_zone"]] = [int(inf), int(sup)]

    return variables, ranges


def prepare_csv_for_2d(in_path, out_path, name="iter"):
    px = pd.read_csv(in_path, index_col=0)
    new_index = []
    for i, j in enumerate(px.index):
        new_index.append(name + "{}".format(i + 1))
    px.index = new_index

    px.to_csv(out_path)


def get_inputs_from_aoc(file_path):
    tree = etree.parse(file_path)
    variables = []
    ranges = {}
    root = tree.getroot()
    for i in root.findall(".//LoiStrickler", root.nsmap):
        lois = i.xpath("@LoiRef")
        if "Fk" in lois[0]:
            lois[0] = lois[0].split("_")[1]
        variables.append(lois[0])
        sup = i.xpath("@Max")
        inf = i.xpath("@Min")
        ranges[lois[0]] = [int(inf[0]), int(sup[0])]
    return variables, ranges


def increase_domain(ranges):
    for key in ranges.keys():
        sup = ranges[key][1] + 1
        inf = ranges[key][0] - 1
        ranges[key] = [inf, sup]
        return ranges


def get_targets_from_aoc(file_aoc, file_lhpt):
    tree_aoc = etree.parse(file_aoc)
    root_aoc = tree_aoc.getroot()
    tree_lhpt = etree.parse(file_lhpt)
    root_lhpt = tree_lhpt.getroot()
    St_names = []
    calcs = []

    for i in root_aoc.findall(".//LoiCalculPermanent", root_aoc.nsmap):
        calcs.append(i.xpath('@CalculRef')[0])
    for i in root_lhpt.findall('.//EchelleSection', root_lhpt.nsmap):
        St_names.append(i.xpath('@SectionRef')[0])
    return [St_names, calcs, 'Z']


def get_observations_from_aoc(file_aoc, file_lhpt):
    tree_aoc = etree.parse(file_aoc)
    root_aoc = tree_aoc.getroot()
    tree_lhpt = etree.parse(file_lhpt)
    root_lhpt = tree_lhpt.getroot()
    cal_names = []
    loi_refs = []
    for i in root_aoc.findall(".//LoiCalculPermanent", root_aoc.nsmap):
        cal_name = i.xpath("@CalculRef")[0]
        loi_ref = i.xpath("@LoiRef")[0]
        cal_names.append(cal_name)
        loi_refs.append(loi_ref)
    obs = {}
    for i in root_lhpt.findall(".//LoiTF", root_lhpt.nsmap):
        loi_ref = i.xpath("@Nom")[0]
        crue = []
        if loi_ref in loi_refs:
            for j in i.findall(".//PointTF", root_lhpt.nsmap):
                crue.append(j.xpath('text()')[0])
            obs[cal_names[loi_refs.index(loi_ref)]] = crue

    values = []
    col_names = []
    for key in obs.keys():
        for i in obs[key]:
            values.append(float(i.split(' ')[-1]))
            col_names.append(key + '/' + i.split(' ')[0])
    df = pd.DataFrame(np.array(values).reshape(1, -1))
    df.columns = col_names
    return obs, df


def generate_px_2d(inname, filecsv, out_folder, sep=";", encoding="utf-8"):
    real_inname = os.path.basename(inname)

    # Ouverture du fichier CSV (traite tout en tant que texte avec dtype=str, sinon le replace derrière bug)

    data = pd.read_csv(filecsv, sep=sep, dtype=str, index_col=0)

    (nrow, ncol) = data.shape

    print("Le fichier {} contient :".format(filecsv))

    print("* {} lignes, correspondant aux fichiers suivantes : {}".format(nrow, list(data.index)))

    print("* {} colonnes, correspondant aux mots-clés suivantes : {}".format(ncol, list(data.columns)))

    # Créé dossier de sortie s'il n'existe pas

    if out_folder is not None:

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # Ouverture du fichier ews d'entrée

    with open(inname, 'r', encoding=encoding) as filein:

        filein_content = filein.readlines()

        # Boucle sur le contenu du fichier CSV

        for i, row in data.iterrows():

            outname = row.name
            new_out_folder = os.path.join(out_folder, outname)
            print("outname", outname)
            if not os.path.exists(new_out_folder):
                os.makedirs(new_out_folder)

            if new_out_folder is not None:
                outname = os.path.join(new_out_folder, real_inname)

            list2replace = row.index
            print(list2replace)

            print("> Export de {}".format(outname))

            # Export du fichier modifie

            with open(outname, 'w', encoding=encoding) as fileout:

                count = {key: 0 for key in data.columns}

                # Boucle sur les mots-clés à remplacer (en-tête du CSV)

                for line in filein_content:

                    for key in list2replace:

                        if key in line:
                            line = line.replace(key, row[key],
                                                1)  # que 1ere occurence (pour consistance avec le conteur)

                            count[key] += 1

                    fileout.write(line)

                # Cherche si le nombre de remplacements est cohérant

                unique_replacement = True

                for (key, value) in count.items():

                    if value is not 1:
                        print("ATTENTION : Le mot-clé {} a été remplacé {} fois".format(key, value))

                        unique_replacement = False

                if unique_replacement:
                    print("Tous les mots-clés ont été remplacés une seule fois")


def copy_file(source_path, target_path, org_file):
    for file in os.listdir(target_path):
        old_name = os.path.join(source_path, org_file)
        new_folder = os.path.join(target_path, file)
        new_name = os.path.join(new_folder, org_file)
        shutil.copyfile(old_name, new_name)


def plot_loi_forttement(ranges, file_opt, out_path):
    df = pd.DataFrame(columns=["Fk", "value", "type"])
    opt_df = pd.read_csv(file_opt)

    for key in ranges.keys():
        fk = key.split('_')[1]
        min_v = ranges[key][0]
        t = "min"
        new_df = pd.DataFrame([[fk, min_v, t]], columns=["Fk", "value", "type"])
        df = df.append(new_df)

        max_v = ranges[key][1]
        t = "max"
        new_df = pd.DataFrame([[fk, max_v, t]], columns=["Fk", "value", "type"])
        df = df.append(new_df)

        opt = opt_df[key].values[0]
        t = "opt"
        new_df = pd.DataFrame([[fk, opt, t]], columns=["Fk", "value", "type"])
        df = df.append(new_df)
    fig = plt.figure(figsize=(30, 15))
    sns.scatterplot(x="Fk", y="value", hue="type", data=df)
    fig.savefig(out_path)


if __name__ == "__main__":
    file_aoc = "./Etu_BV2016A_Conc_Etatref - ISfonds2016_K2016/RS_20200710 17-05-39/BV2016A.aoc_Sans-LE2002-doux.aoc.xml"
    variables, ranges = get_inputs_from_aoc(file_aoc)
    file_opt = "./BV/PX_opt_xgboost/PX_size150_methodeMinDist.csv"
    file_name = os.path.basename(file_opt).split(".")[0]
    path = os.path.dirname(file_opt)
    out_path = os.path.join(path, file_name + ".png")
    plot_loi_forttement(ranges, file_opt, out_path)
