# GUIFold
GUI for running jobs with a local installation of AlphaFold2. Supports submission to queuing systems.

Important note: This is an ALPHA version and development is ongoing. Feedback and bug reports are very welcome.

Main Features:
* Organization of jobs into projects
* Tracking and restoring jobs
* Queue submission with memory estimation and submission script template
* Custom Protein Template
* Disabling MSA and/or template search
* Splitting into feature generation (CPU) and prediction (GPU) steps
* Evaluation pipeline (PAE/pLDDT tables, PAE plots)

![Demo-Image](guifold/images/screenshots.png)


## Installation

The installation requires conda.

```
mkdir guifold
cd guifold
git clone https://github.com/fmi-basel/GUIFold
```

Before proceeding with the installation it is recommended to setup the [global configuration file](#setup-of-global-configuration-file) and [cluster submission script](#setup-of-cluster-submission-template) (if needed).

If you have a separate initialization script for conda (if the initialization is not in your .bashrc) add
```
source path/to/conda_init.sh
```
to the beginning of `guifold/setup_environment.sh`.

Running the setup_environment file will<br/>
* download a [modified alphafold repository](https://github.com/fmi-basel/AF4GUIFold) including some extra features<br/>
* create a conda environment in the same folder<br/>
* install required packages (the packages are listed in the `conda_pkgs.yml` file) <br/>
* install the modified alphafold package<br/>
* install GUIFold<br/><br/>

To start the setup run:
```
bash GUIFold/setup_environment.sh
```
`conda_env` will be the default name of the conda repository. The conda env will be installed with an absolute path (which is also needed for activation).

If you encounter any error try to do the installation step by step.

### Download of genetic databases and params

Follow instructions in the [AlphaFold readme](https://github.com/deepmind/alphafold#genetic-databases).

### Setup of global configuration file

When GUIFold is installed in a shared location it is recommended 
to create a global configuration file so that the users don't have to configure the paths on their own.
When a user starts the app for the first time and a configuration file exists in the expected location, the parameters will be automatically
transferred to the database of the user (stored in the home directory of the user). The user can change settings in the GUI later on.
Open the file `GUIFold/guifold/config/template.conf` and adapt it to your local environment.
Further explanations of the different parameters are given as comments in the file.
After editing, save the file to `GUIFold/guifold/config/guifold.conf`. It is important to use this specific name and location otherwise it will not be loaded.
When the global configuration needs to be changed later on, the users can re-load it in the Settings dialog of the GUI.

Re-install the package if it has been installed before:
```
(conda activate /path/to/af-conda)
cd GUIFold
python setup.py clean --all install clean --all
```


### Setup of cluster submission template

The Jinja2 package is used to render the submission script template. See [Jinja2 documentation](https://jinja.palletsprojects.com/en/3.0.x/) for further information. The variables listed below can be used to create a template. See also examples below.

The template needs to be saved to `GUIFold/guifold/templates/submit_script.j2`. In the same folder you can find an example for a SLURM cluster.

After saving the template to the above location, re-install the package if it has been installed before:
```
(conda activate /path/to/af-conda)
cd GUIFold
python setup.py clean --all install clean --all
```

GUIFold supports the following variables that can be used in the submission template. The parameters are determined dynamically based on the input and settings (configuration file or Settings dialog in the app):<br/><br/>
`{{logfile}}` (required) Path to the log file<br/>
`{{account}}` (optional) When a specific account is needed to run jobs on the cluster<br/>
`{{use_gpu}}` (optional) This can be used to build a conditional (example below) to select CPU or GPU nodes/queues<br/>
`{{gpu_name}}` (optional) If several GPU models are available on the cluster and if they are given in the settings, the app will select the model that has enough memory to run the job. It can also be used in a conditional (see example below).<br/>
`{{mem}}` (optional) How much RAM (in GB) should be reserved. The RAM will be automatically increased with the GPU memory or for unified memory.<br/>
`{{gpu_mem}}` (optional) Useful when the queuing system supports selection of GPU by memory requirement. Value in GB.<br/>
`{{split_mem}}` (optional) If the required memory exceeds the available GPU memory, the job can be run with unified memory. The split_mem variable holds None or the memory split fraction and can be used for a conditional to set the FLAGS required to enable unified memory use (see SLURM example below).<br/>
`{{commnad}}` (required) The command to run the AlphaFold job

To cancel jobs from the GUI the script also needs to write the Job ID to the logfile.
The pattern needs to be as follows:<br/>
```echo "QUEUE_JOB_ID=$JOB_ID_VARIABLE_FROM_QUEUING_SYSTEM"```<br/>
In case of SLURM it would be:<br/>
```echo "QUEUE_JOB_ID=$SLURM_JOB_ID"```

The number of CPUs should be set to 24 (at maximum 2 jackhmmer and 1 hhblits jobs are run in parallel, each set to use 8 CPUs).


#### Example of a template for a SLURM cluster.

The cluster has two types of GPUs, V100 (32 GB) and A100 (80 GB).
When the memory requirement is below 32 GB the scheduler should decide which GPU to use. If the job exceeds 32 GB, only the node with the A100 GPUs is to be used.
Alternatively one could also use `--gres:{{gpu_name}}:1` but it will give less flexibility.

```
#!/bin/bash
#SBATCH --account={{account}}
#SBATCH --job-name=alphafold
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output={{logfile}}
#SBATCH --error={{logfile}}
#Append to logfile
#SBATCH --open-mode=append
{% if use_gpu %}
#SBATCH --partition=gpu_queue
{% else %}
#SBATCH --partition=cpu_queue
{% endif %}
#SBATCH --mem={{mem}}G
{% if use_gpu %}
{% if gpu_name == "a100" %}
#SBATCH --nodelist=a100_node
{% endif %}
#SBATCH --gres=gpu:1
{% endif %}


{% if split_mem %}
export TF_FORCE_UNIFIED_MEMORY=True
export XLA_PYTHON_CLIENT_MEM_FRACTION={{split_mem}}
{% endif %}

module load ... (or conda activate ...)
{{ command }}
```

#### Example of a template for an IBM LSF cluster (NOT TESTED)

```
{% if use_gpu %}
#BSUB -q gpu_queue
{% else %}
#BSUB -q cpu_queue
{% endif %}
#BSUB -n 24                             # 24 cores
#BSUB -W 8:00                           # 8-hour run-time
#BSUB -R "rusage[mem={{mem / 24}}GB]"   # Estimated RAM per core
#BSUB -M {{mem / 24}}GB                 # Maximum RAM per core
#BSUB -J alphafold                      # Jobname
#BSUB -R "span[hosts=1]"                # Run on a single host
#BSUB -o {{logfile}}
#BSUB -e {{logfile}}
{% if use_gpu %}
#BSUB -R "select[gpu_mtotal0>={{gpu_mem}}GB]"
{% endif %}


module load ... (or conda activate ...)
{{ command }}
```

### Setup of a modulefile
Instead of activating the conda env you can also create an [environment modulefile](#https://modules.readthedocs.io/) for production use.

Minimal example:
```
#%Module1.0

setenv       ALPHAFOLD_CONDA            /path/to/guifold/af-conda
prepend-path PATH                       $env(ALPHAFOLD_CONDA)/bin
prepend-path LD_LIBRARY_PATH            $env(ALPHAFOLD_CONDA)/lib
prepend-path LD_LIBRARY_PATH            $env(ALPHAFOLD_CONDA)/x86_64-conda-linux-gnu/sysroot/usr/lib64/
prepend-path PYTHONPATH                 $env(ALPHAFOLD_CONDA)/lib/python3.8/site-packages
prepend-path PYTHONPATH                 $env(ALPHAFOLD_CONDA)/lib/python3.8

```



## Usage

When the conda env is activated (conda activate /path/to/af-conda) or added to PATH/LD_LIBRARY_PATH/PYTHONPATH (see [Setup of a module file](#setup-of-a-module-file) you can start GUIFold by typing:<br/>
`afgui.py`

To re-run an evaluation go to the job folder (where the FASTA sequence is stored) and type<br/>
`afeval.py --fasta_path name_of_sequence.fasta`


## Licenses

GUIFold is licensed under the Apache License, Version 2.0.

Icons are from the GTK framework, licensed under [GPL](https://gitlab.gnome.org/GNOME/gtk/-/blob/main/COPYING).

The modified AlphaFold code retains its original license. See (https://github.com/deepmind/alphafold)

Third-party software and libraries may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.