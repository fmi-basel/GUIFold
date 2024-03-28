#!/usr/bin/bash
set -e
CONDA_ENV_NAME=conda_env
conda_yaml="af_environment.yml"
echo "Starting installation in $(pwd)"
base_dir=$(pwd)
echo "Downloading stereo_chemical_props.txt"
wget --no-check-certificate -P GUIFold/alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
cd $base_dir
echo "Creating conda environment with prefix $(pwd)/$CONDA_ENV_NAME and installing packages"
conda env create -p $(pwd)/$CONDA_ENV_NAME -f GUIFold/${conda_yaml}
conda activate $(pwd)/$CONDA_ENV_NAME
cd $base_dir
fi