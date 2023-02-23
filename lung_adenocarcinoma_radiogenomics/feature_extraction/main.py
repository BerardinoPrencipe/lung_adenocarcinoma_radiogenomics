import subprocess

if __name__ == "__main__":

    dataset = ".\\feature_extraction\\dataset_foggia.csv"
    output_file = "feature_foggia.csv"

    subprocess.run(["pyradiomics", dataset,
                    "-o", output_file,
                    "-f", "csv",
                    "--param", "exampleCT.yaml"])

