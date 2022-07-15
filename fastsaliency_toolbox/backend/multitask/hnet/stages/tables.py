"""
Tables
------

Generates tables (like in the original paper) that show how well the models perform on a variety of image domains.

"""

import os
import pandas as pd
import wandb

from backend.multitask.pipeline.pipeline import AStage


class Tables(AStage):
    model_order = ["AIM", "IKN", "GBVS", "BMS", "IMSIG", "RARE2012", "SUN", "UniSal", "SAM", "DGII"]

    envs = {
        "AIM" : "Matlab",
        "IKN" : "Matlab",
        "GBVS" : "Matlab",
        "BMS" : "Matlab",
        "IMSIG" : "Matlab",
        "RARE2012" : "Matlab",
        "SUN" : "Matlab",
        "UniSal" : "Python",
        "SAM" : "Python",
        "DGII" : "Python",
    }

    mbs = {
        "AIM" : "24MB",
        "IKN" : "1MB",
        "GBVS" : "1MB",
        "BMS" : "5MB",
        "IMSIG" : "20KB",
        "RARE2012" : "20KB",
        "SUN" : "1MB",
        "UniSal" : "30MB",
        "SAM" : "561MB",
        "DGII" : "330MB",
    }

    times_cpu = {
        "AIM" : "10sec",
        "IKN" : "6.0sec",
        "GBVS" : "6.1sec",
        "BMS" : "0.4sec",
        "IMSIG" : "5.7sec",
        "RARE2012" : "6.3sec",
        "SUN" : "12sec",
        "UniSal" : "0.4sec",
        "SAM" : "7.3sec",
        "DGII" : "4.8sec",
    }

    CAT = ["Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal", "Indoor", "Inverted", "Jumbled", "LineDrawing", "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural", "Pattern", "Random", "Satelite", "Sketch", "Social"]
    UMSI = ["ads", "infographics", "mobile_uis", "movie_posters", "webpages"]
    UMSI_table = ["Ads", "Infographics", "MobileUIs", "MoviePosters", "Webpages"]

    def __init__(self, conf, name, verbose):
        super().__init__(name=name, verbose=verbose)
        self._tasks = conf["all_tasks"]
        self._input = None
        
    def setup(self, work_dir_path: str = None, input=None):
        super().setup(work_dir_path, input)
        assert work_dir_path is not None, "Working directory path cannot be None."

        self._input = input
        self._work_dir_path = work_dir_path

        self._data_path = os.path.split(work_dir_path)[0]
    
    def _table_1(self):
        print("TABLE 1")

        data_salicon = pd.read_csv(os.path.join(self._data_path, "table_salicon", "test_results.csv"))
        data_pattern = pd.read_csv(os.path.join(self._data_path, "table_cat_Pattern", "test_results.csv"))

        table_file = os.path.join(self._work_dir_path, "table1.tex")
        with open(table_file, "x") as f:
            f.write("\\begin{table} \n")
            f.write("\\resizebox{\\textwidth}{!}{% \n")
            f.write("   \\begin{tabular}{cccccccc} \n")
            f.write("      \\toprule \n")
            f.write("      Model & \\multicolumn{3}{c}{Original implementation} & \\multicolumn{4}{c}{(Dis)similarity between student \\& original models}                   \\\\\n")
            f.write("      \\cmidrule(r){2-4} \n")
            f.write("      \\cmidrule(r){4-8} \n")
            f.write("       & & & & \multicolumn{2}{c}{Natural Images (SALICON)} & \multicolumn{2}{c}{Patterns (CAT2000)} \\\\ \n")
            f.write("      \\cmidrule(r){5-6} \n")
            f.write("      \\cmidrule(r){7-8} \n")
            f.write("       & Codebase & Size & Time (CPU) & CC $\\uparrow$ & KL $\\downarrow$ & CC $\\uparrow$ & KL $\\downarrow$ \\\\ \n")
            f.write("      \\midrule \n")

            tasks = [t for t in self.model_order if t in self._tasks]
            for task in tasks:
                salicon_row = data_salicon.loc[data_salicon["Model"] == task]
                pattern_row = data_pattern.loc[data_pattern["Model"] == task]

                f.write("      " + task  + " & " + self.envs[task] + " & " + self.mbs[task] + " & " + self.times_cpu[task] + " & \makebox{" \
                    + "%.2f" % salicon_row["CC_mean"].item() + " $\\pm$ " + "%.2f" % salicon_row["CC_std"].item() + "} & \makebox{" \
                    + "%.2f" % salicon_row["KL_mean"].item() + " $\\pm$ " + "%.2f" % salicon_row["KL_std"].item() + "} & \makebox{" \
                    + "%.2f" % pattern_row["CC_mean"].item() + " $\\pm$ " + "%.2f" % pattern_row["CC_std"].item() + "} & \makebox{" \
                    + "%.2f" % pattern_row["KL_mean"].item() + " $\\pm$ " + "%.2f" % pattern_row["KL_std"].item() + "}     \\\\ \n")

            f.write("      \\bottomrule \n")
            f.write("   \\end{tabular} \n")
            f.write("} \n")
            f.write("\\end{table} \n")
        wandb.save(table_file)

    def _table_2(self):
        print("TABLE 2")

        table_file = os.path.join(self._work_dir_path, "table2.tex")
        with open(table_file, "x") as f:
            tasks = [t for t in self.model_order if t in self._tasks]
            f.write("\\begin{table} \n")
            f.write("\\resizebox{\\textwidth}{!}{% \n")
            f.write("   \\begin{tabular}{cccccccccccc} \n")
            f.write("      \\toprule \n")
            f.write("      Dataset & Metric & " + " & ".join(tasks) + " \\\\ \n")
            f.write("      \\midrule \n")
            f.write("      CAT2000 \\\\ \n")
            f.write("      \\midrule \n")

            # CAT
            for i in range(len(self.CAT)):
                myString1 = "      "
                myString1 += "\\multirow{2}{*}{" + self.CAT[i] + "} & CC  $\\uparrow$"
                myTable = pd.read_csv(os.path.join(self._data_path, "table_cat_" + self.CAT[i], "test_results.csv"))
                for task in tasks:
                    row = myTable.loc[myTable["Model"] == task]
                    myString1 += " & \makebox{" +  "%.2f" % row["CC_mean"].item() + " $\\pm$ " + "%.2f" % row["CC_std"].item()  + "}"
                myString1 += " \\\\ \n"
                f.write(myString1)

                myString2 = "      "
                myString2 += " & KL  $\\downarrow$"
                for task in tasks:
                    row = myTable.loc[myTable["Model"] == task]
                    myString2 += " & \makebox{" +  "%.2f" % row["KL_mean"].item() + " $\\pm$ " + "%.2f" % row["KL_std"].item()  + "}"
                myString2 += " \\\\ \n"
                f.write(myString2)

                if i < len(self.CAT) - 1:
                    f.write("      \\cmidrule(r){2-12} \n")


            f.write("      \\midrule \n")

            # UMSI
            f.write("      UMSI \\\\ \n")
            f.write("      \\midrule \n")
            for i in range(len(self.UMSI)):
                myString1 = "      "
                myString1 += "\\multirow{2}{*}{" + self.UMSI_table[i] + "} & CC  $\\uparrow$"
                myTable = pd.read_csv(os.path.join(self._data_path, "table_umsi_" + self.UMSI[i], "test_results.csv"))
                for task in tasks:
                    row = myTable.loc[myTable["Model"] == task]
                    myString1 += " & \makebox{" +  "%.2f" % row["CC_mean"].item() + " $\\pm$ " + "%.2f" % row["CC_std"].item()  + "}"
                myString1 += " \\\\ \n"
                f.write(myString1)

                myString2 = "      "
                myString2 += " & KL  $\\downarrow$"
                for task in tasks:
                    row = myTable.loc[myTable["Model"] == task]
                    myString2 += " & \makebox{" +  "%.2f" % row["KL_mean"].item() + " $\\pm$ " + "%.2f" % row["KL_std"].item()  + "}"
                myString2 += " \\\\ \n"
                f.write(myString2)

                if i < len(self.UMSI) - 1:
                    f.write("      \\cmidrule(r){2-12} \n")

            f.write("      \\bottomrule \n")
            f.write("   \\end{tabular} \n")
            f.write("} \n")
            f.write("\\end{table} \n")
        wandb.save(table_file)

    def execute(self):
        super().execute()

        self._table_1()
        self._table_2()

        return self._input