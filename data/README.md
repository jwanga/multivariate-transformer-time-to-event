# Data Directory

The notebook downloads CMAPSS data into this directory at runtime. Files
here are **not committed** — see the project root `.gitignore`.

## Expected layout after download

```
data/
├── README.md                    # this file
└── CMAPSSData/                  # created by the download cell
    ├── train_FD001.txt
    ├── train_FD002.txt
    ├── train_FD003.txt
    ├── train_FD004.txt
    ├── test_FD001.txt
    ├── test_FD002.txt
    ├── test_FD003.txt
    ├── test_FD004.txt
    ├── RUL_FD001.txt
    ├── RUL_FD002.txt
    ├── RUL_FD003.txt
    ├── RUL_FD004.txt
    ├── Damage Propagation Modeling.pdf
    └── readme.txt
```

The notebook works with FD002 by default but the data loader contract
supports all four subsets.

## Source 1: Kaggle (default — auto-download)

The notebook's download cell uses the Kaggle CLI to fetch
`behrad3d/nasa-cmaps` automatically.

You need a Kaggle API token configured first:

1. Create a Kaggle account at <https://www.kaggle.com>
2. Go to Account → API → Create New API Token
3. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`

Then run the download cell in the notebook.

## Source 2: NASA (manual fallback)

If the Kaggle download fails or you'd rather get the data directly from
the original source:

1. Visit the NASA Open Data Portal entry:
   <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
2. Download the dataset zip
3. Extract its contents into `data/CMAPSSData/` so the files match the
   layout above
4. Re-run the data-loading cell in the notebook (it will skip the
   download step if the files are already present)

## File format

Each `train_FD00X.txt` and `test_FD00X.txt` file is space-separated with
26 columns:

| Column | Name | Description |
|---|---|---|
| 1 | `unit` | Engine identifier (integer) |
| 2 | `cycle` | Time step within this engine's run (integer, starts at 1) |
| 3 | `op_setting_1` | Altitude / operating setting (float) |
| 4 | `op_setting_2` | Mach number / operating setting (float) |
| 5 | `op_setting_3` | Throttle resolver angle / operating setting (float) |
| 6–26 | `sensor_1` … `sensor_21` | Sensor measurements (floats) |

`RUL_FD00X.txt` contains one integer per line — the true remaining cycles
for the corresponding engine in `test_FD00X.txt`. These are the
right-censoring times the survival model uses for evaluation.

For full sensor semantics and degradation modeling background, see
`Damage Propagation Modeling.pdf` inside the dataset.
