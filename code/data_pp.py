import pandas as pd
from pandas import DataFrame
from pathlib import Path

def appendicitis_pp(filepath:Path, save=False):
    """

    :param filepath: Absolute or relative path to the raw Excel data file (.xlsx). Data must be in the first sheet with columns matching the expected variable names from the Data Summary sheet.
    :param save: Dictates whether to save preprocessed data as new excel file or not. Keep as ``false`` when using notebooks
    :return data: Preprocessed dataframe ready for train/test splitting.
    :return descriptors: Dataframe containing preprocessed descriptors sheet.

    Preprocesses the pediatric appendicitis dataset for use in binary classification models predicting appendicitis diagnosis. See ``class-balancing-np.ipynb`` for interactive notebook code.
    """

    # Load in data set
    xlsx = pd.ExcelFile(filepath)
    raw_data = pd.read_excel(xlsx, 0)
    descriptors = pd.read_excel(xlsx, 1)

    # Filter out observations missing labels
    data = raw_data.dropna(subset=['Diagnosis'])

    # Filter out observations where US_Performed is unknown or no
    data = data.dropna(subset=['US_Performed'])
    data = data[data['US_Performed'] != 'no']
    data = data.drop(['US_Number', 'US_Performed'], axis=1)

    # Drop retrospective variables
    data = data.drop(['Length_of_Stay', 'Management', 'Severity'], axis=1)

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

    data['Appendix_Wall_Layers'] = data['Appendix_Wall_Layers'].map(wall_layer_map)

    # Appendicolith Findings (-1 if measurement missing)
    appendicolith_map = {
        'yes': 1,
        'suspected': 1,
        'no': 0,
    }

    data['Appendicolith'] = data['Appendicolith'].map(appendicolith_map)

    # Perfusion Findings
    perfusion_map = {
        'no': 0,  # absent — most concerning
        'hypoperfused': 1,  # reduced — concerning
        'present': 2,  # normal
        'hyperperfused': 3,  # increased — inflammation
    }

    data['Perfusion'] = data['Perfusion'].map(perfusion_map)

    # Perforation Findings (-1 if measurement missing)
    perforation_map = {
        'no': 0,  # no perforation
        'not excluded': 1,  # cannot rule out
        'suspected': 2,  # likely perforated
        'yes': 3,  # confirmed perforated
    }

    data['Perforation'] = data['Perforation'].map(perforation_map)

    # Abscess Findings (-1 if measurement missing)
    abscess_map = {
        'no': 0,
        'suspected': 1,
        'yes': 1,
    }

    data['Appendicular_Abscess'] = data['Appendicular_Abscess'].map(abscess_map)

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

    data['Gynecological_Findings'] = data['Gynecological_Findings'].map(gynae_map)

    # Deal with the rest of the categorical variables
    # Sex number by alphabetical order
    sex_map = {
        'female': 0,
        'male': 1,
    }
    data['Sex'] = data['Sex'].map(sex_map)
    data = data.dropna(subset=['Sex'])

    # Ketones
    ketone_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['Ketones_in_Urine'] = data['Ketones_in_Urine'].map(ketone_map)

    # RBC
    RBC_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['RBC_in_Urine'] = data['RBC_in_Urine'].map(RBC_map)

    # WBC
    WBC_map = {
        'no': 0,
        '+': 1,
        '++': 2,
        '+++': 3,
    }

    data['WBC_in_Urine'] = data['WBC_in_Urine'].map(WBC_map)

    # Stool (abnormal = 1)
    stool_map = {
        'normal': 0,
        'constipation': 1,
        'diarrhea': 1,
        'constipation, diarrhea': 1,
    }

    data['Stool'] = data['Stool'].map(stool_map)

    # Peritonitis
    peritonitis_map = {
        'no': 0,
        'local': 1,
        'generalized': 2,
    }

    data['Peritonitis'] = data['Peritonitis'].map(peritonitis_map)

    # Psoas_Sign
    psoas_map = {
        'no': 0,
        'yes': 1,
    }

    data['Psoas_Sign'] = data['Psoas_Sign'].map(psoas_map)

    # Change diagnosis to binary
    diagnosis_map = {
        'appendicitis': 1,
        'no appendicitis': 0,
    }

    data['Diagnosis'] = data['Diagnosis'].map(diagnosis_map)

    # Break into classes
    data = assign_classes(data)

    # Check that all non-NaN are numerical
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
    test_all_non_nan_values_numeric(data)

    if descriptors['Variable Group'].isna().any():
        descriptors['Variable Group'] = descriptors['Variable Group'].ffill()

    print('Preprocessing Done')

    # Save pre-processed dataframe to excel file
    if save:
        data.to_excel('../code/app_data_pp.xlsx', index=False)

    return data, descriptors

def assign_classes(data: pd.DataFrame):
    """
    Assign classes to each observation using preprocessed pediatric appendicitis dataset

    :param data: Preprocessed Pediatric Appendicitis dataset
    :return:
    """

    # Create new column to store class
    data['Class'] = 0

    # Map presumed appendicitis -> 1, no appendicitis -> 0
    appendicitis_labels = [
        'appendicitis',
        'Appendizitis, Lymphadenitis mesenterialis',
        'Appendizitis/ Lymphadenitis mesenterialis',
        'chronische Appendizitis',
        'Sepsis mit Begleitappendizitis',
        'gedeckt perforierte Appendizitis',
    ]

    data['Diagnosis_Presumptive'] = data['Diagnosis_Presumptive'].apply(
        lambda x: 1 if x in appendicitis_labels else 0
    )

    # Map management and diagnosis to four different classes
    class_map = {
    (0, 0): 0, # presumed no app, no appendicitis -> 0

    (0, 1): 1, # presumed no app, appendicitis -> 1

    (1, 0): 2, # presumed yes app, no appendicitis -> 2

    (1, 1): 3, # presumed yes app, appendicitis -> 3
    }

    data['Class'] = data.apply(
        lambda row: class_map[(row['Diagnosis_Presumptive'], row['Diagnosis'])], axis=1
    )

    return data

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    data_path = (base_dir / '..' / 'data' / 'app_data.xlsx').resolve()

    data, descriptors = appendicitis_pp(data_path)
    print(data.head())
    print(descriptors.head())

