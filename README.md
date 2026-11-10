# TabUL Rasa: Vertical Tabular Unlearning

Implementation of the CIKM 2026 short paper:

> **Toward TabUL Rasa in Tabular Diffusion Models**

[[Paper (ACM OpenAccess)]](https://dl.acm.org/doi/10.1145/3799682.3839881)

TabUL Rasa studies *vertical* (feature/column-level) unlearning in tabular diffusion models: removing the
influence of a single column from a pre-trained generative backbone. The code builds on
[TabSyn](https://github.com/amazon-science/tabsyn) and
[TabDiff](https://github.com/MinkaiXu/TabDiff), and adds a collection of unlearners on top of them.

## Contents

- [Installation](#installation)
- [Workflow overview](#workflow-overview)
- [Dataset preparation](#dataset-preparation)
- [Pre-training a TabDiff backbone](#pre-training-a-tabdiff-backbone)
- [Unlearning a column](#unlearning-a-column)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation

Python 3.10 is recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

pip install -e .   # editable install, for development
# or
pip install .      # regular install, if you only want to use the code
```

## Workflow overview

Running an experiment consists of three steps:

1. **Prepare** (download and process) your dataset(s).
2. **Pre-train** a TabDiff backbone on a dataset.
3. **Unlearn** a specific column from the pre-trained backbone.

> [!IMPORTANT]
> Internally, columns are reordered into a canonical layout of contiguous blocks:
> 1. numerical columns, 2. integer columns, 3. categorical columns, 4. the target column.
>
> Check that your data layout matches your expectations, especially when selecting the column to unlearn. Relevant code:
> - Dataset metadata in [`data/Info`](data/Info)
> - [`src/tabdiff/datasets/processors`](src/tabdiff/datasets/processors), in particular
    >   `preprocess` in [`preprocessors.py`](src/tabdiff/datasets/processors/preprocessors.py) and
    >   `recover_data` in [`postprocessors.py`](src/tabdiff/datasets/processors/postprocessors.py).

### Configuration with Hydra

All experiments are configured through [Hydra](https://hydra.cc/), so whole grids of experiments can be launched
with multirun (`-m`) instead of shell scripts. Configurations live in [`configs/`](configs).

## Dataset preparation

### Datasets used in the paper

```bash
python examples/download_dataset.py   # download raw datasets
python examples/process_dataset.py    # process them
```

### Using your own dataset

1. Create a directory for your dataset and place your data in it as `data/<DATASET>/<DATASET>.csv`.
   **The first row must be a header** with the column names; the remaining rows are records.

2. Create a metadata file `data/Info/<DATASET>.json`:

   ```jsonc
   {
       "name": "<DATASET>",
       "task_type": "binclass",          // "binclass" or "regression"
       "header": "infer",
       "column_names": null,
       "num_col_idx": [/* indices of numerical columns */],
       "cat_col_idx": [/* indices of categorical columns */],
       "target_col_idx": [/* indices of target column (used for MLE) */],
       "file_type": "csv",
       "data_path": "data/<DATASET>/<DATASET>.csv",
       "test_path": null
   }
   ```

3. Process the dataset:

   ```bash
   python examples/process_dataset.py --dataname <DATASET>
   ```

## Pre-training a TabDiff backbone

Train an unconditional TabDiff model on the full table.
Supported datasets: `adult`, `default`, `shoppers`, `magic`, `beijing`, `news`.
The training configuration is in [`configs/tabdiff.yaml`](configs/tabdiff.yaml).

To train several datasets in one go with a Hydra sweep (by default `adult`, `default`, `shoppers` and `magic`):

```bash
python3 train_model.py -m --config-name tabdiff_train_sweep

# Restrict the sweep to a subset of datasets
python3 train_model.py -m --config-name tabdiff_train_sweep dataset=adult,shoppers,magic
```

## Unlearning a column

Once a backbone is trained, run `untrain_model.py` with the config of the unlearner you want to use:

```bash
python3 untrain_model.py --config-name tabdiff_unlearn_sweep_<UNLEARNER>
```

| `<UNLEARNER>`          | Method                                               |
|------------------------|------------------------------------------------------|
| `feature_ascent`       | Gradient Ascent                                      |
| `feature_diff`         | Gradient Ascent + Gradient Descent on retain data    |
| `feature_npo`          | Negative Preference Optimization (vertical)          |
| `feature_npo_baseline` | Negative Preference Optimization                     |
| `feature_orth`         | Score Orthogonal                                     |
| `feature_shuffle`      | Unlearning with a shuffled column                    |

For example, Gradient Ascent uses
[`configs/tabdiff_unlearn_sweep_feature_ascent.yaml`](configs/tabdiff_unlearn_sweep_feature_ascent.yaml).

Unlearning runs also evaluate the selected metrics. To speed this up, increase
`experiment.evaluation_parallelism` (e.g. `experiment.evaluation_parallelism=10`), but keep it at or below the
number of logical cores on your machine.

## Evaluation

> Note that for evaluating unlearning, refer to the Unlearning section above.

Evaluation uses the same Hydra configuration as training:

```bash
# Quality (SDMetrics) and DCR experiments on all sweep datasets
python3 examples/eval_quality.py -m --config-name tabdiff_train_sweep experiment=quality,dcr

# Quality only, on a subset of datasets
python3 examples/eval_quality.py -m --config-name tabdiff_train_sweep experiment=quality dataset=news,default,magic
```

To evaluate a single model, point to its config and (optionally) set the dataset explicitly:

```bash
python3 eval_quality.py --config-name tabdiff dataset.name=<DATASET>
```

This includes the additional fidelity metrics ($\alpha$-precision and $\beta$-recall); make sure samples have
been generated first.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{galjaard2026tabulrasa,
   author = {Galjaard, Jeroen M. and Zhu, Chaoyi and Birke, Robert and Chen, Pin-Yu and Chen, Lydia Y.},
   title = {Toward TabUL Rasa in Tabular Diffusion Models},
   year = {2026},
   isbn = {9798400725395},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3799682.3839881},
   doi = {10.1145/3799682.3839881},
   booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
   pages = {7101–7105},
   location = {Rome, Italy},
   series = {CIKM '26}
}
```

## Acknowledgements

This repository builds on [TabSyn](https://github.com/amazon-science/tabsyn) and
[TabDiff](https://github.com/MinkaiXu/TabDiff). Many thanks to Hengrui Zhang and the TabDiff authors for making
their code available.

## License

This project is released under the [MIT License](LICENSE).
