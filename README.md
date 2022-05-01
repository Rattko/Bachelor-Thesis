# Bachelor Thesis at Charles University

## Description - TODO

## Quick Preview

You can use [Docker](https://www.docker.com) for a quick preview of the functionality. Run the following command:

```zsh
docker run --rm -it -p 5001:5001 rattko/bachelor-thesis:latest
```

The command downloads a pre-built [docker image](https://hub.docker.com/r/rattko/bachelor-thesis) from [Docker Hub](https://hub.docker.com) and runs it. The docker image fires up a Mlflow server and executes an experiment consisting of two preprocessing methods over two datasets. Four runs will be performed in total, each lasting roughly 90 seconds. One of the runs should fail due to insufficient training time; the other three may or may not finish successfully, depending on your PC's computing power. You can observe the results of the runs using Mlflow UI accessible on `127.0.0.1:5001`. Mlflow server will continue running after the experiment has finished until you stop the container or press `Ctrl-C`.

## Installation

Unfortunately, one of our core dependencies does not support Windows operating system. Thus, we are forced to require either macOS or Linux operating system. We also use the new syntax for type hints requiring Python 3.10. Furthermore, [AutoSklearn](https://automl.github.io/auto-sklearn/master/index.html) requires [SWIG](https://www.swig.org). It can be installed using Homebrew on macOS or any package manager on Linux. Once these requirements have been met, we can obtain the source code and proceed with the installation. Run the following commands:

```zsh
git clone git@github.com:Rattko/Bachelor-Thesis.git && cd Bachelor-Thesis
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r tools/requirements.txt
pip3 install -e .
```

We also need to patch AutoSklearn to gain complete control over the preprocessing steps in the experiment.

```zsh
bash tools/patch_autosklearn.sh
```

## Running Experiments

Once we have completed the installation process described in the previous section, we can proceed to run an experiment or two. First, we need to boot up a Mlflow tracking server using:

```zsh
mlflow server --backend-store-uri sqlite:///.mlruns.db --default-artifact-root .mlruns-artifacts &> .mlflow.logs &
```

We redirect the server's `stdout` and `stderr` to a log file and boot it in the background so that we can continue using the same terminal window. This command boots up a server accessible on `127.0.0.1:5000` in the browser. Once we have run an experiment, information about runs will appear on that address in real-time.

Now we need to download some datasets to use in experiments. We can use

```zsh
./tools/download_openml_datasets.py
```

for this. The script automatically downloads datasets satisfying pre-specified conditions from [OpenML](https://www.openml.org). A progress bar will pop up, showing the download status. After a successful download, we are ready to execute an experiment. Run the following command:

```zsh
python3 src/core/main.py \
    --datasets 310 40900 \
    --preprocessings smote tomek_links \
    --total_time 90 \
    --time_per_run 30
```

You can consult the help page of the script to learn more about the various supported switches. However, the following should be sufficient to get you up to speed. The `--datasets` switch expects the names of the datasets as found in the `datasets/` directory without any extension. Likewise, the `--preprocessing` switch expects the names of the preprocessing methods found in the `src/core/preprocessings/` directory. The switch also accepts special values to run `all`, only `oversampling` or only `undersampling` preprocessing methods. The last two switches control the time allocation for model training in AutoSklearn. See `time_left_for_this_task` and `per_run_time_limit` on [this&nbsp;link](https://automl.github.io/auto-sklearn/master/api.html) for an explanation.
