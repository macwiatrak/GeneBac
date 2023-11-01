# GeneBac

[![genebac_overview](files/images/genebac_overview.png)]()

GeneBac is a modular framework for predicting antibiotic resistance in bacteria from DNA sequence.
It takes as input the DNA sequence of selected genes with variants integrated into the sequence and outputs the predicted MIC to a set of drugs.
GeneBac can also be used for other tasks, including variant effect scoring, gene expression prediction, resistant genes identification, and strain clustering.

This repository contains the training and code for the GeneBac model described in "[Sequence-based modelling of bacterial genomes enables accurate antibiotic resistance prediction](input_text)".

## Installation

GeneBac is implemented in Python `3.9`. To setup with the repo with all dependencies, clone the repository locally and create
an environment, for example with `virtualenwrapper` or `conda`. To install the dependencies, navigate to the directory 
of the repository and run:
```bash
pip install .
```

## Antibiotic resistance prediction
To train the GeneBac model for antibiotic resistance prediction, run:
```bash
python deep_bac/train_gene_pheno.py --input-dir <input_dir> --output-dir <output_dir> 
```
We provide a processed [CRyPTIC _Mycobacterium Tuberculosis_](insert_link) dataset of `12,460` strains for training and evaluation on 
[Google drive](insert_link) (5 GB). The folder contains processed DNA sequences with variants integrated into the sequence as well as 
 training, test and cross-validation splits.

To evaluate the GeneBac model for antibiotic resistance prediction, run:
```bash
python deep_bac/train_gene_pheno.py --test --input-dir <input_dir> --output-dir <output_dir> --ckpt-path <path_to_the_trained_model_checkpoint>
```

## Variant effect scoring
To compute variant effect scores for a single variant, run:

```bash
import ...


# load the trained model
model = load_trained_pheno_model(ckpt_path=<path_to_the_trained_model_checkpoint>)

# compute the variant effect size
variant_effect_size = compute_variant_effect_size(
    model=model,
    gene=<gene_name>,
    variant=<variant>,
    start_idx=<start_idx>,
    end_idx=<end_idx>
)
```
It is also possible to compute variant effect scores for a batch of variants. 
To do it you firstly need to create a file with variants. An example file is provided in `files/variants.tsv`.
Then, run:
```bash
python deep_bac/experiments/variant_scoring/run_ism.py ...
```
## Gene Expression prediction
To train the GeneBac model for gene expression prediction, run:
```bash
python deep_bac/train_gene_expr.py --input-dir <input_dir> --output-dir <output_dir> ...
```
We provide a processed gene expression dataset of `396` _Pseudomonas aeruginosa_ strains 
on [Google drive](insert_link_here), where each gene in a strain is a separate example. 
The folder contains processed DNA sequences with variants integrated into the sequence as well as 
 training, validation and test splits.

## Resistant genes identification
GeneBac can also be used to identify genes that are associated with antibiotic resistance to particular drugs.
We compute the association using the [DeepLift](https://arxiv.org/abs/1704.02685) algorithm. To compute the gene-drug scores
on a test set using a trained model, run:
```bash
python deep_bac/experiments/loci_importance/run_loci_importance.py --input-dir <input_dir> --output-dir <output_dir> --ckpt-path <path_to_the_trained_model_checkpoint>
```
Here, the `<input-dir>` can be the same as the one used for antibiotic resistance prediction.
## Strain clustering

## Checkpoints

## Citations
If you find GeneBac useful in your work, please cite our paper:
```bibtex
```

## License
GeneBac is licensed under the [MIT License](https://opensource.org/license/mit/).
