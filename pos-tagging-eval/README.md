## Part-of-Speech Experiments

This folder contains scripts for reproducing results in Tables 2 and 9 in [the paper](https://arxiv.org/pdf/2406.13560).

### Requirements

For completing the pipeline, you first need to run the `../intrinsic-segmentation-eval/Snakefile` pipeline - this is done automatically in the local `Snakefile`, but you need to make sure you have all the requirements ready.
On top of the in intrinsic evaluation requirements, you need to `pip install -r requirements.txt` from this directory as well.

### Running the pipeline

After installing prerequisites, running can be as simple as 

```bash
Snakemake -c$jobs
```

where `$jobs` is the number of concurrent jobs to run. We recommend using a scheduling system and profile settings which would observe the `resources` specifications in the snakemake rules. For running a single experiment with the tagger, we recommend using a 16GB GPU.
