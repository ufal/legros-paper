# Machine Translation Experiments

## Prerequisities

Python requirements:
 * `datasets`
 * `morfessor`
 * `sentencepiece`
 * `snakemake`
 * `sacrebleu`

Having compiled
 * `sentencepiece`
 * `marian`

## Running the experiments

The experiment directories can be prepared by calling

```bash
python3 prepare_iwslt_2017.py
```

The script downloads the IWSLT 2017 datasets and prepares the experiment
directories with a Snake file and Marian config.

The experiments can be executed by running `snakemake` in the respective
experiment directories.

## What happens during an experiment

