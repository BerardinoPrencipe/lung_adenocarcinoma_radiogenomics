import os

import pandas as pd


def preprocess_foggia_dataset(df_features, df_labels, target):

    images_path = df_features["Image"]
    features_patients_id = [d.split("\\")[-1].split(".")[0] for d in images_path]
    df_features["Image"] = features_patients_id
    df_features = df_features.rename(columns={"Image": "Patient ID"})
    df_labels = df_labels.drop(["Histotype", "TC", "Segmentation"], axis=1)
    if target == "multi":
        df_labels_target = df_labels[["Patient ID", "Mut_KRAS", "Mut_EGFR"]]
    elif target == "KRAS":
        df_labels_target = df_labels[["Patient ID", "Mut_KRAS"]]
    elif target == "EGFR":
        df_labels_target = df_labels[["Patient ID", "Mut_EGFR"]]

    df_labels_target.dropna(inplace=True)
    df_features_target = df_features.join(df_labels_target.set_index("Patient ID"), on="Patient ID")
    df_features_target.drop(["Unnamed: 0"], axis=1, inplace=True)
    # df_features_target.dropna(inplace=True)

    print(df_features_target)
    return df_features_target



def preprocess_NSCLC_dataset(df_features, df_labels, target):
    pass


def preprocess_dataset(df_features, df_labels, target="KRAS", df_type="foggia"):

    if df_type == "foggia":
        return preprocess_foggia_dataset(df_features, df_labels, target)
    if df_type == "NSCLC":
        return preprocess_NSCLC_dataset(df_features, df_labels, target)

