import pandas as pd
import random
import os

import pandas as pd
import argparse

def generate_splits(split_number, train_ratio, clinical_data_path, save_path):
    # Read the clinical data
    slides = pd.read_csv(clinical_data_path, sep='\t')

    # Assuming 'submitter_id' is the patient ID and 'event' is the column indicating event occurrence (1 for event, 0 for no event)
    tcga_ids = slides['submitter_id']
    events = slides['event']

    data = pd.DataFrame({'submitter_id': tcga_ids, 'event': events})

    event_patients = data[data['event'] == 1]
    non_event_patients = data[data['event'] == 0]

    # Ensure we have an equal number of event and non-event patients in a split
    min_patients = min(len(event_patients), len(non_event_patients))
    event_patients = event_patients.sample(min_patients, random_state=42)
    non_event_patients = non_event_patients.sample(min_patients, random_state=42)

    balanced_data = pd.concat([event_patients, non_event_patients])

    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    num_train = int(train_ratio * len(balanced_data) // 2)
    num_test = (len(balanced_data) // 2) - num_train

    split_data = pd.DataFrame(index=balanced_data['submitter_id'])

    # Generate the splits
    for i in range(1, split_number + 1):
        train_event = event_patients.sample(num_train, random_state=i)
        test_event = event_patients.drop(train_event.index).sample(num_test, random_state=i)

        train_non_event = non_event_patients.sample(num_train, random_state=i)
        test_non_event = non_event_patients.drop(train_non_event.index).sample(num_test, random_state=i)

        train_ids = pd.concat([train_event, train_non_event])['submitter_id']
        test_ids = pd.concat([test_event, test_non_event])['submitter_id']

        # Create the split column with 'Train' and 'Test' labels
        split_labels = ['Train'] * len(train_ids) + ['Test'] * len(test_ids)
        ids = pd.concat([train_ids, test_ids])

        # Shuffle the split data
        temp_split = pd.DataFrame({'id': ids, 'label': split_labels})
        temp_split = temp_split.sample(frac=1, random_state=i).reset_index(drop=True)

        split_data[str(i)] = temp_split.set_index('id')['label']

    split_data = split_data.rename_axis('SLIDE_ID').reset_index()

    print(split_data.head())

    split_data.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data splits for TCGA slides.")
    parser.add_argument('--split_number', type=int, default=5, help="Number of splits to generate")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Training dataset ratio")
    parser.add_argument('--clinical_data_path', type=str, required=True, help="Path to read clinical data file")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the generated splits CSV file")

    args = parser.parse_args()

    generate_splits(args.split_number, args.train_ratio, args.clinical_data_path, args.save_path)

