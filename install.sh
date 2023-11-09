#!/usr/bin/bash
set -e
CONDA_ENV_NAME=conda_env
if [[ "$@" =~ "no-rosettafold" && "$@" =~ "no-fastfold" ]]; then
    conda_yaml="af_environment.yml"
elif [[ "$@" =~ "no-fastfold" ]]; then
    conda_yaml="af_rf_environment.yml"
elif [[ "$@" =~ "no-rosettafold" ]]; then
    conda_yaml="af_ff_environment.yml"
else
    conda_yaml="af_ff_rf_environment.yml"
fi
echo "Starting installation in $(pwd)"
base_dir=$(pwd)
echo "Downloading stereo_chemical_props.txt"
wget --no-check-certificate -P GUIFold/alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
cd $base_dir
echo "Creating conda environment with prefix $(pwd)/$CONDA_ENV_NAME and installing packages"
conda env create -p $(pwd)/$CONDA_ENV_NAME -f GUIFold/${conda_yaml}
conda activate $(pwd)/$CONDA_ENV_NAME
if ! [[ "$@" =~ "no-fastfold" ]]; then
echo "Installing fastfold"
git clone https://github.com/hpcaitech/FastFold.git
cd FastFold
git checkout add_iptm_in_heads
python setup.py install
cd $base_dir
fi
cd ${base_dir}/GUIFold
python setup.py install
cd alphafold
python setup.py install
cd $base_dir
if ! [[ "$@" =~ "no-rosettafold" ]]; then
echo "Installing rosettafold"
cd GUIFold/rosettafold
python setup.py install
cd rosettafold/SE3Transformer
python setup.py install
cd $base_dir
fi