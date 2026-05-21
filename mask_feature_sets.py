import pandas as pd
from data_pp import appendicitis_pp

def mask_feature_sets(X_data: pd.DataFrame, descriptors: pd.DataFrame):
    """
    Filters X_data columns by variable group to produce three feature sets
    for clinical, clinical+lab, and clinical+lab+ultrasound models.

    :param X_data: Preprocessed feature dataframe with all variables.
    :param descriptors: Data Summary sheet dataframe containing at minimum
                        ``Variable Group`` and ``Variable Name in Data Files``
                        columns. For example: ``pd.read_excel('app_data.xlsx', sheet_name='Data Summary')``.
    :return: Tuple of three dataframes (X_clinical, X_lab, X_us).
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    # Forward fill merged cells in Variable Group column
    if descriptors['Variable Group'].isna().any():
        descriptors['Variable Group'] = descriptors['Variable Group'].ffill()

    # Build group → column name mapping
    variable_groups = ['Demographic / Other', 'Clinical', 'Scoring', 'Laboratory', 'Ultrasound']
    variable_dict = {group: [] for group in variable_groups}

    for group in variable_groups:
        variables = descriptors[descriptors['Variable Group'] == group]
        variables = variables['Variable Name in Data Files'].tolist()
        variable_dict[group] = variables

    # Define cumulative column sets per model
    clinical_cols = variable_dict['Demographic / Other'] + variable_dict['Clinical']
    lab_cols      = clinical_cols + variable_dict['Scoring'] + variable_dict['Laboratory']
    us_cols       = lab_cols + variable_dict['Ultrasound']

    # Filter to only columns present in X_data (guards against dropped columns)
    def filter_cols(cols):
        return X_data[[col for col in cols if col in X_data.columns]]

    X_data_clinical = filter_cols(clinical_cols)
    X_data_lab      = filter_cols(lab_cols)
    X_data_us       = filter_cols(us_cols)

    return X_data_clinical, X_data_lab, X_data_us


# Usage
if __name__ == '__main__':
    xlsx = pd.ExcelFile('data/app_data.xlsx')
    descriptors = pd.read_excel(xlsx, 1)
    X = appendicitis_pp('data/app_data.xlsx')

    X_clinical, X_lab, X_us = mask_feature_sets(X, descriptors)

    print(X_clinical.columns)
    print(X_lab.columns)
    print(X_us.columns)