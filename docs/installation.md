
# Installation guide
The `arborist` package can be installed either via [conda (recommended)](#installation-via-conda-recommended) or [manually](#manual-installation) via `pip`.  Both methods will automatically install [dependencies](#dependencies)

## Installation via conda (recommended)

`arborist` is available on Bioconda. You can easy install it via `conda/mamba` with:

```bash
conda install -c bioconda -c conda-forge arborist
```

## Manual installation
Clone the repository and install the package using `pip`. *Note: dependencies will be installed automatically*


```bash
git clone https://github.com/VanLoo-lab/Arborist.git
cd Arborist
pip install .

```

## Testing the installation

To test the installation of the CLI tool and Python package, run the following:
```bash
arborist -h
python -c "from arborist import arborist"
```


## Dependencies

The `arborist` package requires the following dependencies:

- `python>=3.7`
- `numpy>=1.20`
- `scipy>=1.7`
- `pandas>=1.3`
- `pygraphviz`
- `numba>=0.61`
- `networkx>=3.6`
