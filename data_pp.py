import pandas as pd
from pandas import DataFrame


def appendicitis_pp(filepath: str) -> DataFrame:
    """

    :param filepath: Absolute or relative path to the raw Excel data file (.xlsx). Data must be in the first sheet with columns matching the expected variable names from the Data Summary sheet.
    :return data: Preprocessed dataframe ready for train/test splitting.

    Preprocesses the pediatric appendicitis dataset for use in binary classification models predicting appendicitis diagnosis and surgical management. See ``class-balancing-np.ipynb`` for interactive notebook code.
    """

    # Load in data set
    xlsx = pd.ExcelFile(filepath)
    raw_data = pd.read_excel(xlsx, 0)

    # Filter out observations missing labels
    data = raw_data.dropna(subset=['Diagnosis'])

    # Filter out observations where US_Performed is unknown or no
    data = data.dropna(subset=['US_Performed'])
    data = data[data['US_Performed'] != 'no']
    data = data.drop(['US_Number', 'US_Performed'], axis=1)

    # Drop retrospective variables
    data = data.drop(['Length_of_Stay', 'Management', 'Severity', 'Diagnosis_Presumptive'], axis=1)

    # Segmented_Neutrophils is pretty sparsely populated, redundant with neutrophil percent
    data = data.drop(['Segmented_Neutrophils'], axis=1)

    # Convert binary yes/no columns --> 1 or 0
    # All binary yes/no columns from the Data Summary sheet
    binary_cols = [
        # Clinical
        'Migratory_Pain',
        'Lower_Right_Abd_Pain',
        'Contralateral_Rebound_Tenderness',
        'Ipsilateral_Rebound_Tenderness',
        'Coughing_Pain',
        'Nausea',
        'Loss_of_Appetite',
        'Dysuria',
        # Lab
        'Neutrophilia',
        # Ultrasound
        'Appendix_on_US',
        'Free_Fluids',
        'Target_Sign',
        'Surrounding_Tissue_Reaction',
        'Pathological_Lymph_Nodes',
        'Bowel_Wall_Thickening',
        'Ileus',
        'Coprostasis',
        'Meteorism',
        'Enteritis',
        'Conglomerate_of_Bowel_Loops',
    ]

    # Loop: convert yes = 1, no = 0, NaN = -1
    yes_no_map = {'yes': 1, 'no': 0}

    for col in binary_cols:
        if col in data.columns:
            data[col] = (
                data[col]
                .map(yes_no_map)  # maps yes/no, leaves NaN as NaN
                .fillna(-1)  # NaN = -1
                .astype(int)
            )

    # One-hot encoding for categorical ultrasound features (non-binary features)
    # Parameters: Appendix_Wall_Layers, Target_Sign, Appendicolith, Perfusion, Perforation, Surrounding_Tissue_Reaction, Appendicular_Abscess, Abscess_Location, Pathological_Lymph_Nodes, Lymph_Nodes_Location, Bowel_Wall_Thickening, Conglomerate_of_Bowel_Loops, Ileus, Coprostasis, Meteorism, Enteritis, Gynecological_Findings

    # Find unique terms
    # parameters = ['Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith', 'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess', 'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location', 'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus', 'Coprostasis', 'Meteorism', 'Enteritis', 'Gynecological_Findings']

    # for parameter in parameters:
    #     print(f'{parameter} unique: {data[parameter].unique()}')

    data = data.drop(['Abscess_Location', 'Lymph_Nodes_Location'], axis=1)

    # Wall Layer Findings (-1 if measurement missing)
    wall_layer_map = {
        'intact': 0,  # normal
        'raised': 1,  # mild — layers lifted but present
        'partially raised': 1,  # mild — same severity as raised
        'upset': 2,  # severe — layers disrupted
    }

    data['Appendix_Wall_Layers'] = data['Appendix_Wall_Layers'].map(wall_layer_map).fillna(-1)

    # Appendicolith Findings (-1 if measurement missing)
    appendicolith_map = {
        'yes': 1,
        'suspected': 1,
        'no': 0,
    }

    data['Appendicolith'] = data['Appendicolith'].map(appendicolith_map).fillna(-1)

    # Perfusion Findings
    perfusion_map = {
        'no': 0,  # absent — most concerning
        'hypoperfused': 1,  # reduced — concerning
        'present': 2,  # normal
        'hyperperfused': 3,  # increased — inflammation
    }

    data['Perfusion'] = data['Perfusion'].map(perfusion_map).fillna(-1)

    # Perforation Findings (-1 if measurement missing)
    perforation_map = {
        'no': 0,  # no perforation
        'not excluded': 1,  # cannot rule out
        'suspected': 2,  # likely perforated
        'yes': 3,  # confirmed perforated
    }

    data['Perforation'] = data['Perforation'].map(perforation_map).fillna(-1)

    # Abscess Findings (-1 if measurement missing)
    abscess_map = {
        'no': 0,
        'suspected': 1,
        'yes': 1,
    }

    data['Appendicular_Abscess'] = data['Appendicular_Abscess'].map(abscess_map).fillna(-1)

    # Gynecological Findings - 1 if abnormal finding present, 0 if normal/absent, -1 if measurement missing (nan)
    gynae_map = {
        'Ovarialzyste': 1,  # ovarian cyst
        'Ovarialzyste ': 1,
        'Ovarialzyste re.': 1,
        'kleine Ovarzyste rechts': 1,
        'Ovarialzysten': 1,
        'Zyste Uterus': 1,  # uterine cyst
        'In beiden Ovarien Zysten darstellbar, links Ovar mit regelrechter Perfusion, rechts etwas vergrößert, keine eindeutige Perfusion nachweisbar. Retrovesikal freie Flüssigkeit mit Binnenecho': 1,
        'V. a. Ovarialtorsion': 1,  # suspected ovarian torsion
        'ja': 1,  # ambiguous but likely abnormal
        'Ausschluss pathologischer Ovarialbefund': 0,  # pathological finding excluded
        'Ausschluss gyn. Ursache der Beschwerden': 0,  # gynae cause excluded
        'kein Anhalt für eine gynäkologische Ursache der Beschwerden': 0,  # no gynae cause
        'unauffällig': 0,  # normal
        'keine': 0,  # none
    }

    data['Gynecological_Findings'] = data['Gynecological_Findings'].map(gynae_map).fillna(-1)

    # Deal with the rest of the categorical variables
    # Sex number by alphabetical order
    sex_map = {
        'female': 0,
        'male': 1,
    }
    data['Sex'] = data['Sex'].map(sex_map)
    data = data.dropna(subset=['Sex'])  # only drops 1 observation

    # Ketones
    ketone_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['Ketones_in_Urine'] = data['Ketones_in_Urine'].map(ketone_map).fillna(-1)

    # RBC
    RBC_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['RBC_in_Urine'] = data['RBC_in_Urine'].map(RBC_map).fillna(-1)

    # WBC
    WBC_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['WBC_in_Urine'] = data['WBC_in_Urine'].map(WBC_map).fillna(-1)

    # Stool (abnormal = 1)
    stool_map = {
        'normal': 0,
        'constipation': 1,
        'diarrhea': 1,
        'constipation, diarrhea': 1,
    }

    data['Stool'] = data['Stool'].map(stool_map).fillna(-1)

    # Peritonitis
    peritonitis_map = {
        'no': 0,
        'local': 1,
        'generalized': 2,
    }

    data['Peritonitis'] = data['Peritonitis'].map(peritonitis_map).fillna(-1)

    # Psoas_Sign
    psoas_map = {
        'no': 0,
        'yes': 1,
    }

    data['Psoas_Sign'] = data['Psoas_Sign'].map(psoas_map).fillna(-1)

    # Change diagnosis to binary
    diagnosis_map = {
        'appendicitis': 1,
        'no appendicitis': 0,
    }

    data['Diagnosis'] = data['Diagnosis'].map(diagnosis_map)

    # Check that there are no NaN type in diagnosis, check that all non-NaN are numerical

    def test_diagnosis_binary_no_nan(df):
        """Test that all values in 'labels' are 0 or 1 with no NaN."""
        assert df['Diagnosis'].isnull().sum() == 0, \
            f"Found {df['Diagnosis'].isnull().sum()} NaN values in 'Diagnosis'"

        invalid = ~df['Diagnosis'].isin([0, 1])
        assert invalid.sum() == 0, \
            f"Found non-binary values in 'Diagnosis': {df.loc[invalid, 'Diagnosis'].unique()}"

        print("All diagnosis labels are 0 or 1 with no NaN")

    def test_all_non_nan_values_numeric(df):
        """Test that all non-NaN values across the dataframe are numeric (int or float)."""
        non_numeric_cols = []

        for col in df.columns:
            non_nan_values = df[col].dropna()
            if not pd.api.types.is_numeric_dtype(non_nan_values):
                non_numeric_cols.append(col)

        assert len(non_numeric_cols) == 0, \
            f"Non-numeric values found in columns: {non_numeric_cols}"

        print("All non-NaN values are numeric")

    # Run tests
    test_diagnosis_binary_no_nan(data)
    test_all_non_nan_values_numeric(data)

    print('Preprocessing Done')

    return data

if __name__ == '__main__':
    data = appendicitis_pp('data/app_data.xlsx')
    print(data.head())