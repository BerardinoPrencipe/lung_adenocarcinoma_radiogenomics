import pandas as pd


def extract_radiomics_features_with_labels(dataset_df, label_df, target="KRAS"):

    label_df_id = label_df["Case ID"]

    diagnostic_info_cols = [el for el in dataset_df.columns if 'diagnostics_' in el]
    dataset_df_postprocessed = dataset_df.drop(columns=['Image', 'Mask', *diagnostic_info_cols])

    return dataset_df_postprocessed

