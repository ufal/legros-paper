#!/usr/bin/env python3

import datasets
import os
import logging

logging.basicConfig(level=logging.INFO)


def process_dataset(name, src2, src3, tgt2, tgt3):
    dir_path = f"{name}/plain_data"
    logging.info("Processing '%s' into '%s'.", name, dir_path)
    os.makedirs(dir_path, exist_ok=True)

    logging.info("Downloading dataset.")
    dataset = datasets.load_dataset('iwslt2017', f'iwslt2017-{src2}-{tgt2}')
    logging.info("Lowercase and save.")
    for split in ['train', 'validation', 'test']:
        split_out = split
        if split == 'validation':
            split_out = 'dev'
        with open(f"{dir_path}/{split_out}.{src3}", 'w') as f:
            for line in dataset[split]['translation']:
                print(line[src2].lower(), file=f)
        with open(f"{dir_path}/{split_out}.{tgt3}", 'w') as f:
            for line in dataset[split]['translation']:
                print(line[tgt2].lower(), file=f)

    logging.info("Snake file for the experiment.")
    with open(f"{name}/languages.py", 'w') as f:
        print(f"LNG1 = '{src3}'", file=f)
        print(f"LNG2 = '{tgt3}'", file=f)

    logging.info("Linking Snakefile and config.yml into the experiment directory.")
    os.symlink('../config.yml', f"{name}/config.yml")
    os.symlink('../Snakefile', f"{name}/Snakefile")
    logging.info("Done.")


process_dataset('iwslt2017-de-en', 'de', 'deu', 'en', 'eng')
process_dataset('iwslt2017-en-fr', 'en', 'eng', 'fr', 'fra')
process_dataset('iwslt2017-ar-en', 'ar', 'ara', 'en', 'eng')
process_dataset('iwslt2017-ro-it', 'ro', 'ron', 'it', 'ita')
process_dataset('iwslt2017-ro-nl', 'ro', 'ron', 'nl', 'nld')
process_dataset('iwslt2017-it-nl', 'it', 'ita', 'nl', 'nld')
process_dataset('iwslt2017-it-en', 'it', 'ita', 'en', 'eng')
process_dataset('iwslt2017-en-nl', 'en', 'eng', 'nl', 'nld')
process_dataset('iwslt2017-en-ro', 'en', 'eng', 'ro', 'ron')
