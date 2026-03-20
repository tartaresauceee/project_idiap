
# test_victor

Loads and processes Victor `.zarr` data, then plots state and wrench signals.

## Prerequisites
- Miniforge or Conda installed
- Shell initialized for Conda (`conda init zsh`)

## Setup
```bash
conda env create -f environment.yml
conda activate exportData
```

## Run
```bash
python victor_io_zarr.py
```

## Data path
`victor_io_zarr.py` currently uses a hard-coded `zarr_root` path in `main()`.
Update that path to your local dataset.

