CONDA_ENV_NAME=conda_env
base_dir=$(pwd) && \
git clone https://github.com/fmi-basel/AF4GUIFold && \
cd AF4GUIFold && \
git checkout guifold && \
wget --no-check-certificate -P AF4GUIFold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
cd $base_dir && \
conda env create -p $(pwd)/$CONDA_ENV_NAME -f GUIFold/conda_pkgs.yml
conda activate $(pwd)/$CONDA_ENV_NAME && \
pip install --upgrade pip && \
pip install jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
cd $(pwd)/$CONDA_ENV_NAME/lib/python3.8/site-packages && \
patch -p0 < $base_dir/AF4GUIFold/docker/openmm.patch && \
cd $base_dir && \
cd AF4GUIFold && \
python setup_local.py install && \
cd $base_dir && \
cd GUIFold && \
python setup.py install