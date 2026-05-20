import pandas as pd

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
    from data_pp import appendicitis_pp
    # appendicitis_pp uses assign_classes
    data = appendicitis_pp('data/app_data.xlsx')
    print(data['Diagnosis_Presumptive'])
    print(data['Diagnosis'])
    print(data['Class'])