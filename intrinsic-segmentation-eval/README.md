# Initrinsic evaluation of the tokenizer

### Prerequisites

`pip install -r requirements.txt`

## Prepare training data

Preparing training includes (1) downlowding News Crawl (or CC-100 for
Mongolian), (2) tokenizing and lowercasing the data and (3) training FastText
embedding on that data.

```bash
bash donwnload_data.sh ${LNG}
```

## Prepare test data

We use two test sets for intrinsic evaluation:

* Universal Segmentations.
* Test data from SIGMORPHON 2022 shared task.

Data in universal segmentations come from various sources and they are to a
large extent automatically generated. To get a reasonable language coverage, we
split the language vocabulary into deciles and sample 1000/K words from K-th
decile into the test set.

The test data from the SIGMORPHON 2022 shared task are manually annotated ant
therefore they should be of better quality. Here, the issue is that in most
languages, the data contains morpheme instead of morphs. We use a simple
algorithm to match morphemes to the surface form segments based on longest
common substring. This algorithm fails for approx. 4% of the words, which we
discard. We only sample 5k words per language to speedup the evaluation.

```bash
bash compile_test_data.sh
```

## Baselines

As a baseline for segmetnation, we use BPE, Unigram LM from SentencePiece and
Morphessor.

Scripts:

* `submit_bpe_train.sh`

* `submit_spm_train.sh`

* `submit_morfessor_train.sh`

To submit training all baselines on a SLURM cluster, run
```bash
train_baselines_on_slurm.sh
```

Baselines are need to initialize the actual experiments.

## Experiments

For our experiments, we initialize the tokenization with Unigram SentencePiece
and BPE.

The first step is to create experiment directories which contain the initial
subword segmentation.

```bash
bash prepare_experiments.sh
```

After than the `run_experiments.sh` script could be used to run the actual
experiments. It assumes the directory structure from the previous steps and has
the following arguments:

* `-l` or `--language`

* `-s` or `--size` for initial SentencePiece or BPE size

* `-i` or `--init` for initialization type: one of BPE of SentencePiece

To submit all experiments on SLURM, call
```bash
bash submit_all_experiments.sh
```

Running the experiment does not include the tokenizer evalauation. Run

```bash
bash eval_all_experiments.sh
```

for continuous evaluation of experiments that already finished experiments.
