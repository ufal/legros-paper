# Initrinsic evaluation of the tokenizer

### Prerequisites

1. Build `legros` and its prerequisites
2. Build `sentencepiece` inside the `3rd_party` directory at the root of this
 repository 

`pip install -r requirements.txt`

### Automated evaluation pipeline using Snakemake

After installing the requirements, make sure to configure Snakemake to
work with your cluster (see its documentation to figure out how - Snakemake
supports Slurm backend, which we make use of in the pipeline (see the 
`resources` property of every rule))

Once everything is set up, consult the rule `all` in `Snakefile` - by default
it is set to run **all** the intrinsic evaluation experiments. With a CPU 
cluster containing a few 64-core servers with 256GB RAM the whole pipeline
takes a few days to finish (not counting debugging time). Most of the time
the 64-core machines are not necessary (for example, Morfessor does not 
support parallel processing so everything in this phase runs in a single 
thread)

To run the pipeline, simply run the following line (with an appropriate limit
for concurrent jobs - for example, we use 64):

```bash
snakemake -c64
```

At the end, the `score`, `recall` and `fscore` files should appear with the 
results similar to the ones in our paper. There is randomness in how you shuffle
the data before trimming it on 50M sentences.
