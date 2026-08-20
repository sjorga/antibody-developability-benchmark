# Antibody Developability Benchmark

*A reproducible benchmark comparing physicochemical descriptors, protein language model embeddings, and structure-derived surface features for antibody developability prediction within a common modeling and evaluation framework.*

**Highlights:** 4 antibody representations · 2 developability endpoints · 2 regression models · cluster-aware cross-validation

**Core tools:** Python · scikit-learn · pandas · ANARCI · ESM2 · ABodyBuilder2 · FreeSASA

---

## Scientific Motivation

Antibody developability prediction requires information about each antibody to be represented numerically before it can be used as input to a predictive model. Different computational representations capture different types of molecular information, ranging from compact physicochemical descriptors to high-dimensional protein language model embeddings and structure-derived surface features. Representation choice may therefore influence predictive performance, and its usefulness may depend on the developability property being predicted.

In practice, however, representations are often evaluated as part of larger predictive pipelines in which the model, optimization strategy, and validation procedure may also differ. When several components change simultaneously, it becomes difficult to determine how much of the observed performance difference can be attributed to the representation itself.

This benchmark isolates representation as the primary experimental variable by comparing four antibody representations using the same GDPa1 dataset, Ridge and Elastic Net models, cluster-aware cross-validation folds, and evaluation metrics. Because experimentally characterized antibody developability datasets are typically small, the benchmark also examines whether higher-dimensional representations provide a predictive advantage over compact, interpretable representations under these small-data conditions, and whether relative representation performance differs between developability endpoints.

The reproducible Python pipeline constructs the representations, trains and evaluates the models, and generates the analysis outputs in a single execution.

---

## Repository Structure

```text
antibody-developability-benchmark/
├── data/
│   └── raw/                         # Place the GDPa1 dataset here
├── notebooks/
│   └── antibody_developability_benchmark.ipynb
├── results/
│   └── final/
├── figures/
│   └── final/
├── tables/
│   └── final/
├── run_pipeline.py
├── environment.yml
├── LICENSE
├── README.md
└── .gitignore
```

---

## Benchmark Design

### Dataset

* GDPa1 antibody developability benchmark
* 246 antibodies; 242 measurements available for each modeled endpoint
* Hydrophobic interaction chromatography (HIC)
* Affinity-capture self-interaction nanoparticle spectroscopy (AC-SINS, pH 7.4)

### Antibody Representations

* Global physicochemical descriptors
* Region-aware physicochemical descriptors
* Protein language model embeddings (`esm2_t6_8M_UR50D`)
* Structure-derived surface descriptors (ABodyBuilder2 + FreeSASA)

### Predictive Models

* Ridge regression
* Elastic Net regression

### Evaluation

* Five-fold cluster-aware cross-validation
* Mean fold-level performance for the primary comparison
* Pooled out-of-fold performance as a validation analysis
* Spearman correlation for rank-based performance
* Root mean squared error (RMSE) for prediction error

---

## Key Findings

* Representation performance was endpoint-dependent.
* Structure-derived descriptors achieved the highest mean predictive performance for HIC.
* Global physicochemical descriptors achieved the highest mean predictive performance for AC-SINS.
* Higher-dimensional learned embeddings did not consistently outperform the smaller descriptor sets under the evaluated conditions.
* The leading representation for each endpoint was consistent across Ridge and Elastic Net regression.
* Several representations showed relatively similar mean performance and substantial variation across cross-validation folds, so the observed rankings should not be overinterpreted.

---

## Installation

```bash
git clone https://github.com/sjorga/antibody-developability-benchmark.git
cd antibody-developability-benchmark
conda env create -f environment.yml
conda activate antibody-developability-benchmark
```

The pipeline has been validated using Python 3.10 through the provided Conda environment.

---

## Dataset Access

The GDPa1 dataset is not distributed with this repository. Obtain it from the official Ginkgo Datapoints GDPa1 dataset page and comply with its access conditions and terms of use:

https://huggingface.co/datasets/ginkgo-datapoints/GDPa1

Download `GDPa1_v1.2_20250814.csv` and place it at:

```text
data/raw/GDPa1_v1.2_20250814.csv
```

---

## Running the Pipeline

Run the complete benchmark with:

```bash
python run_pipeline.py
```

The pipeline generates or loads the four feature representations, trains and evaluates Ridge and Elastic Net models using cluster-aware cross-validation, performs coefficient-stability analyses, and produces the figures, tables, predictions, and performance summaries.

---

## Outputs

Generated outputs include:

* Feature matrices
* Fold-level metrics and out-of-fold predictions
* Mean-fold and pooled out-of-fold performance comparisons
* Model coefficients and coefficient-stability analyses
* Benchmark figures and summary tables
* Supplementary analyses

Outputs are written to the `results/`, `figures/`, and `tables/` directories. Validated deliverables are stored in the corresponding `final/` subdirectories.

---

## License

This project is released under the MIT License. See `LICENSE` for the full license text.
