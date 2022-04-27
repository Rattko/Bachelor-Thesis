# Bachelor Thesis at Charles University

## Description - TODO

## Docker Image - TODO

## Installation

Unfortunately, one of our core dependencies does not support Windows operating system, and thus, we are forced to require either macOS or Linux operating system. The installation requires at least Python 3.10 as we use the new type hints syntax introduced in that version. Furthermore, AutoSklearn requires a `swig` library which you can install using Homebrew on macOS or any package manager on Linux. Once these requirements have been met, we can obtain the source code and proceed with the installation. Run the following commands:

 - `git clone git@github.com:Rattko/Bachelor-Thesis.git && cd Bachelor-Thesis`
 - `python3 -m venv .venv && source .venv/bin/activate`
 - `pip3 install -r tools/requirements.txt`
 - `pip3 install -e .`
