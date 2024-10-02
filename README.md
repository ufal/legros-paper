## Lexically Grounded Subword Segmentation: Experimental Code

This repository accompanies the paper "Lexically Grounded Subword Segmentation" by Jindřich Libovický and Jindřich Helcl.
If you find yourself inspired by this, please cite the following paper (watch out for TODOs before November 2024):

```bibtex
@inproceedings{libovicky-helcl-2024-lexically,
  title="Lexically Grounded Subword Segmentation",
  author = "Libovick{\'y}, Jind{\v{r}}ich and Helcl, Jind{\v{r}}ich",
  editor = "TODO",
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2024",
  address = "Miami, USA",
  publisher = "Association for Computational Linguistics",
  url = "TODO",
  doi = "TODO",
  pages = "TODO"
}
```


### Steps to reproduce results

Here follow the instructions for the replication of the paper results.

#### Prerequisites

In order to do everything like us, make sure you have the following tools installed:

* SentencePiece - if you don't want to use your own SentencePiece (which means
  editing multiple files with the right location), go to
  `3rd_party/sentencepiece` and build the tool inside the `build` subfolder.
  The `spm_*` executables will be located in
  `3rd_party/sentencepiece/build/src` in the end.

* Snakemake
