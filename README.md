# RDsynthesisML
This repository contains `python` scripts and notebooks for replicating the results in "Towards Autonomous Materials Synthesis via Reaction-Diffusion Coupling".

## Installation
To use the `python` scripts and notebooks, it is recommended to install `python=3.10` and the packages in [`requirements.txt`](requirements.txt) in a `conda` environment:

```bash
conda install rd_synth_ml python=3.10
pip install -r requirements.txt
```

## Useage
- [`0_scalarizer.ipynb`](0_scalarizer.ipynb): Requires [`scalarizer.py`](scalarizer.py), [`step_detect.py`](step_detect.py), and [`102808_67_2025-02-16_0333.Jpg`](102808_67_2025-02-16_0333.Jpg)
- [`1_eda.ipynb`](1_eda.ipynb): Requires [`history.csv`](history.csv)
- [`2_ml.ipynb`](2_ml.ipynb): Requires [`history.csv`](history.csv)
- [`3_bo.ipynb`](3_bo.ipynb): Requires [`acq.py`](acq.py) and [`history.csv`](history.csv)
- [`4_sample.ipynb`](4_sample.ipynb): Requires [`acq.py`](acq.py), [`history.csv`](history.csv), and [`new_data.csv`](new_data.csv)