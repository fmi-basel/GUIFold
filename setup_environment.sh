CONDA_ENV_NAME=conda_env
base_dir=$(pwd) && \
wget --no-check-certificate -P GUIFold/alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
cd $base_dir && \
conda env create -p $(pwd)/$CONDA_ENV_NAME -f GUIFold/conda_pkgs.yml
conda activate $(pwd)/$CONDA_ENV_NAME && \
pip install --upgrade pip && \
pip install jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
cd $(pwd)/$CONDA_ENV_NAME/lib/python3.8/site-packages && \
patch -p0 < $base_dir/GUIFold/alphafold/docker/openmm.patch && \
cd $base_dir && \
cd GUIFold && \
python setup.py install && \
cd alphafold && \
python setup.py install && \
cd $base_dir