CONDA_ENV_NAME=conda_env
base_dir=$(pwd) && \
wget --no-check-certificate -P GUIFold/alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
cd $base_dir && \
conda env create -p $(pwd)/$CONDA_ENV_NAME -f GUIFold/af_ff_environment.yml
conda activate $(pwd)/$CONDA_ENV_NAME && \
cd $(pwd)/$CONDA_ENV_NAME/lib/python3.8/site-packages && \
patch -p0 < $base_dir/GUIFold/alphafold/docker/openmm.patch && \
cd $base_dir && \
git clone https://github.com/hpcaitech/FastFold.git && \
cd FastFold && \
python setup.py install && \
cd $base_dir && \
cd GUIFold && \
python setup.py install && \
cd alphafold && \
python setup.py install && \
cd $base_dir