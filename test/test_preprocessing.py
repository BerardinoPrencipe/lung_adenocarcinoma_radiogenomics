import pandas as pd
from preprocessing.dataset_preprocessing import preprocess_dataset


def test_preprocessing():

    dataset_foggia = "/Users/berardinoprencipe/Library/CloudStorage/Dropbox/Noduli_Polmonari/Dataset/Local_Foggia/dataset_anonimo/dataset_foggia_anonimo.xlsx"
    feature_foggia = "../dataset/features_foggia.csv"
    dataset_NSCLC = "/Users/berardinoprencipe/Library/CloudStorage/Dropbox/Noduli_Polmonari/Dataset/Public_NSCLC_Radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
    feature_NSCLC = "../dataset/features_NSCLC.csv"

    df_labels_foggia = pd.read_excel(dataset_foggia, engine='openpyxl')
    df_labels_nsclc = pd.read_csv(dataset_NSCLC)
    df_features_foggia = pd.read_csv(feature_foggia)
    df_features_nsclc = pd.read_csv(feature_NSCLC)

    df_features_target_foggia = preprocess_dataset(df_features_foggia, df_labels_foggia, target="KRAS", df_type="foggia")
    df_features_target_foggia = preprocess_dataset(df_features_foggia, df_labels_foggia, target="EGFR", df_type="foggia")
    df_features_target_foggia = preprocess_dataset(df_features_foggia, df_labels_foggia, target="multi", df_type="foggia")
    df_features_target_nsclc = preprocess_dataset(df_features_nsclc, df_labels_nsclc, target="KRAS", df_type="NSCLC")
    df_features_target_nsclc = preprocess_dataset(df_features_nsclc, df_labels_nsclc, target="EGFR", df_type="NSCLC")
    df_features_target_nsclc = preprocess_dataset(df_features_nsclc, df_labels_nsclc, target="multi", df_type="NSCLC")


