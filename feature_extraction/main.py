import subprocess
import pandas as pd


if __name__ == "__main__":

    dataset = "dataset_NSCLC.csv"
    output_file = "../dataset/features_NSCLC.csv"

    dataset_pd = pd.read_csv(dataset, index_col=0)

    # for _, el in dataset_pd.iterrows():
    #
    #     print("Processing Image {}".format(el[0]))
    #
    #     subprocess.run(["pyradiomics", el[0], el[1],
    #                     "-o", output_file,
    #                     "-f", "csv",
    #                     "--param", "exampleCT.yaml",
    #                     "--validate"])

    subprocess.run(["pyradiomics", dataset,
                    "-o", output_file,
                    "-f", "csv",
                    "--param", "exampleCT.yaml"])

