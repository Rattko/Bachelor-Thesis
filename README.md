# Bachelor Thesis at Charles University

## Description - TODO

## Docker Image - TODO

## Installation

Unfortunately, one of our core dependencies does not support Windows operating system, and thus, we are forced to require either macOS or Linux operating system. The installation requires at least Python 3.10 as we use the new type hints syntax introduced in that version. Furthermore, AutoSklearn requires a `swig` library which you can install using Homebrew on macOS or any package manager on Linux. Once these requirements have been met, we can obtain the source code and proceed with the installation. Run the following commands:

 - `git clone git@github.com:Rattko/Bachelor-Thesis.git && cd Bachelor-Thesis`
 - `python3 -m venv .venv && source .venv/bin/activate`
 - `pip3 install -r tools/requirements.txt`
 - `pip3 install -e .`

## Running Experiments Locally

Once we have completed the installation process described in the previous section, we can proceed to run an experiment or two. First, we need to boot up a Mlflow tracking server using:

```zsh
mlflow server --backend-store-uri sqlite:///.mlruns.db --default-artifact-root .mlruns-artifacts &> .mlflow.logs &
```

We redirect the server's `stdout` and `stderr` to a log file and boot it in the background so that we can continue using the same terminal window. This command boots up a server accessible on `127.0.0.1:5000` in the browser. Once we run an experiment, various information about the run will appear in Mlflow and can be viewed in the browser in real-time.

Now we need to download some datasets to use in experiments. We can use the included script `download_openml_datasets.py` in the `tools` directory for this. This script automatically downloads datasets satisfying conditions pre-specified in the script from OpenML. You can run `./tools/download_openml_datasets.py --help` to learn about possible options, optionally adjust them to your liking, and finally run the script using `./tools/download_openml_datasets.py`. A progress bar will pop up, showing the download status. After a successful download, we are ready to execute an experiment. Run the following command:

```zsh
python3 src/core/main.py \
    --datasets 4135 4154 \
    --preprocessings smote tomek_links \
    --total_time 120 \
    --time_per_run 50
```

Again, you can consult the help page of a script to learn more about the various supported switches, but the following should be sufficient to get you up to speed. The `--datasets` switch expects the names of the datasets as found in the `datasets/` directory without any extension. Likewise, the `--preprocessing` switch expects the names of the preprocessing methods found in the `src/core/preprocessings/` directory. The switch also accepts special values to run `all`, only `oversampling` or only `undersampling` preprocessing methods.
