# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import random
import zipfile
from multiprocessing import Pool, Manager, Value
import preprocess
import json
from sklearn.model_selection import StratifiedKFold
import time

# Shared counters for progress tracking
successful_files = 0
error_files = 0
start_time = 0

# Define process_file outside run_gen_genre
def process_file(file_name):
    global successful_files, error_files
    result = preprocess.G(file_name)
    if result is True:
        successful_files += 1
    elif result is False:
        error_files += 1
    
    # Print progress
    processed = successful_files + error_files
    if processed % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Processed: {processed} files - Success: {successful_files}, Errors: {error_files}")
    
    return result


def run_gen_genre():
    global start_time, successful_files, error_files
    
    # Reset counters
    successful_files = 0
    error_files = 0
    
    # Initialize multiprocessing in preprocess
    preprocess.init_multiprocessing()
    
    subset = input('subset: ')
    raw_data_dir = subset + '_data_raw'
    if os.path.exists(raw_data_dir):
        print('Output path {} already exists!'.format(raw_data_dir))
        sys.exit(0)

    data_path = input('LMD dataset zip path: ')
    n_folds = 5
    n_times = 4
    max_length = int(input('sequence length: '))

    preprocess.sample_len_max = max_length
    preprocess.deduplicate = False
    preprocess.data_zip = zipfile.ZipFile(data_path)
    fold_map = dict()
    
    # Create Manager and shared list here, after multiprocessing is initialized
    manager = Manager()
    all_data = manager.list()
    pool_num = 8  # adjustable

    labels = dict()
    with open('midi_genre_map.json') as f:
        for s in json.load(f)[subset].items():
            labels[s[0]] = tuple(
                sorted(set(i.strip().replace(' ', '-') for i in s[1])))

    def get_id(file_name):
        return file_name.split('/')[-1].split('.')[0]

    def get_fold(file_name):
        return fold_map[get_id(file_name)]

    def get_sample(output_str_list):
        max_len = max(len(s.split()) for s in output_str_list)
        return random.choice([s for s in output_str_list if len(s.split()) == max_len])

    def new_writer(file_name, output_str_list):
        if len(output_str_list) > 0:
            all_data.append((file_name, tuple(get_sample(output_str_list)
                                              for _ in range(n_times))))

    preprocess.writer = new_writer

    os.makedirs(raw_data_dir, exist_ok=True)
    print("Finding MIDI files in the archive...")
    file_list = [file_name for file_name in preprocess.data_zip.namelist()
                 if file_name.lower().endswith(('.mid', '.midi'))]
    
    print(f"Found {len(file_list)} MIDI files total. Filtering for those with genre labels...")
    file_list = [file_name for file_name in file_list if get_id(file_name) in labels]
    print(f"Found {len(file_list)} MIDI files with genre labels in the mapping file.")
    
    random.shuffle(file_list)

    print("Creating cross-validation folds...")
    label_list = ['+'.join(labels[get_id(f)]) for f in file_list]
    fold_index = 0
    for train_index, test_index in StratifiedKFold(n_folds).split(file_list, label_list):
        for i in test_index:
            fold_map[get_id(file_list[i])] = fold_index
        fold_index += 1

    # Progress tracking for MIDI processing
    total_files = len(file_list)
    print(f"\nProcessing {total_files} MIDI files with genre labels...")
    start_time = time.time()
    
    # Process files with progress tracking
    with Pool(pool_num) as p:
        results = list(p.map(process_file, file_list))

    # Calculate and display processing statistics
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete in {elapsed_time/60:.1f} minutes.")
    print(f"Results: {successful_files} successful, {error_files} errors, "
          f"{total_files - successful_files - error_files} skipped.")

    random.shuffle(all_data)
    print('{}/{} ({:.2f}%) MIDI files successfully processed.'.format(
        len(all_data), len(file_list), len(all_data) / len(file_list) * 100))

    print("\nCreating dataset files for each fold...")
    for fold in range(n_folds):
        os.makedirs(f'{raw_data_dir}/{fold}', exist_ok=True)
        preprocess.gen_dictionary(f'{raw_data_dir}/{fold}/dict.txt')

        for cur_split in ['train', 'test']:
            prefix = f'{raw_data_dir}/{fold}/{cur_split}'
            with open(prefix + '.txt', 'w') as f_txt, \
                 open(prefix + '.label', 'w') as f_label, \
                 open(prefix + '.id', 'w') as f_id:
                count = 0
                for file_name, output_list in all_data:
                    if (cur_split == 'train' and fold != get_fold(file_name)) or \
                       (cur_split == 'test' and fold == get_fold(file_name)):
                        for i in range(n_times if cur_split == 'train' else 1):
                            f_txt.write(output_list[i] + '\n')
                            f_label.write(' '.join(labels[get_id(file_name)]) + '\n')
                            f_id.write(get_id(file_name) + '\n')
                            count += 1
                print(f"Fold {fold}, Split {cur_split}: {count} examples")
    
    print("\nDataset preparation complete! Next step: binarize the dataset with:")
    print(f"bash binarize_genre.sh {subset}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    run_gen_genre()