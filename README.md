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
* Support of MMseqs2/colabfold MSA pipeline (in addition to original MSA pipeline)
* Automated pairwise interaction screening (with parallelisation over multiple GPUs)
* Evaluation pipeline (PAE/pLDDT tables, PAE plots)

**See ![Wiki](https://github.com/fmi-basel/GUIFold/wiki/Usage) for more detailed documentation.**

![Demo-Image](guifold/images/screenshots.png)

## Installation

### Installation of the repository and dependencies

The installation requires conda.

```
mkdir guifold
cd guifold
git clone --recurse-submodules https://github.com/fmi-basel/GUIFold
```

It is important to clone with the `--recurse-submodules` option to include the modified alphafold module.

Before proceeding with the installation it is recommended to setup the [global configuration file](#setup-of-global-configuration-file) and [cluster submission script](#setup-of-cluster-submission-template) (if needed).

If you have a separate initialization script for conda (if the initialization is not in your .bashrc) add
```
source path/to/conda_init.sh
```
to the beginning of `GUIFold/install.sh`.

Running the install.sh file will<br/>
* create a conda environment in the same folder<br/>
* install required packages (the packages are listed in the `_environment.yml` files <br/>
* install the modified alphafold package<br/>
* install GUIFold<br/><br/>

To run the setup:
```
bash GUIFold/install.sh
```
`conda_env` will be the default name of the conda repository. The conda env will be installed with an absolute path (which is also needed for activation).

(Optional) Install [MMseqs2](https://github.com/soedinglab/mmseqs2) to use the colabfold protocols for MSA generation.

Test if the GUI opens (see Troubleshooting):
```
conda activate /path/to/conda_env
afgui.py
```

#### Troubleshooting

Issue: When trying to run `afgui.py`: `ImportError: libXss.so.1: cannot open shared object file: No such file or directory`
Solution: Add the following path to the LD_LIBRARY_PATH in the command prompt: `export LD_LIBRARY_PATH=/path/to/conda_env/x86_64-conda-linux-gnu/sysroot/usr/lib64/:$LD_LIBRARY_PATH`


### Download of genetic databases and params

Follow instructions in the [AlphaFold readme](https://github.com/deepmind/alphafold#genetic-databases).


### (OPTIONAL) Setup of global configuration file

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
(conda activate /path/to/conda_env)
cd GUIFold
python setup.py clean --all install clean --all
```

### (OPTIONAL) Setup of a local configuration file

As an alternative to a global configuration file stored inside the GUIFold installation directory, a "local" configuration file can be created from the template `GUIFold/guifold/config/template.conf` and stored in any location. To load the settings from this file go to the `Settings` dialog in the GUI and click the `Load local config` button (bottom right). 

### Manual configuration of settings inside the GUI

If no configuration file is supplied, settings need to be defined in the `Settings` dialog and will be stored in the user's database. This has to be done by each user separately.

### (OPTIONAL) Setup of MMseqs2 and colabfold databases

The following steps are only required if the optional feature pipeline 'colabfold_local' is intended to be used. It is not necessary to run the default alphafold pipeline.

#### MMSeqs2 installation

MMSeqs is not automatically installed. It can be easily added to the conda environment with
```
(conda activate /path/to/conda_env)
conda install -c conda-forge -c bioconda mmseqs2
```
or installed in diffrent ways as described in [MMseqs2 documentation](https://mmseqs.com/latest/userguide.pdf)

#### Database setup

If you want to use the colabfold protocol you also need to download "uniref30_2202" and "colabfold_envdb_202108" (available at [Link](https://colabfold.mmseqs.com/)). It is required to convert these databses to expandable profile databases and generate database indices (see [MMseqs2 documentation](https://mmseqs.com/latest/userguide.pdf)).

`mmseqs createindex` is used to create database indices. The `--split 0` flag will automatically determine the number of splits based on the available RAM. Therefore this step should be run on the machine where Alphafold is later run (or adjusted to the minimal available RAM with `--split-memory-limit` in addition to the `--split 0` flag). `--threads` can be used for parallelisation. More details in [MMseqs2 documentation](https://mmseqs.com/latest/userguide.pdf).


1. Go to the uniref90 database directory (which contains uniref90.fasta) and run
```
mmseqs createindex uniref90 tmp --split 0
```

In the global configuration file (see below) the uniref90_mmseqs path needs to point to `your_directory_with_databases/uniref90/uniref90`

2. Go to the uniprot database directory (which contains uniprot.fasta) and run
```
mmseqs createindex uniprot tmp --split 0
```

In the global configuration file the uniref90_mmseqs path needs to point to 
`your_directory_with_databases/uniprot/uniprot`

3. Go to the colabfold_envdb directory and run
```
mmseqs tsv2exprofiledb colabfold_envdb_202108 colabfold_envdb_202108_db
mmseqs createindex colabfold_envdb_202108_db tmp --split 0
```

In the global configuration file the colabfold_envdb path needs to point to 
`your_directory_with_databases/colabfold_envdb_202108_db/colabfold_envdb_202108_db`

4. Go to the uniref30_2202 directory and run
```
mmseqs tsv2exprofiledb uniref30_2202 uniref30_2202_db
mmseqs createindex uniref30_2202_db tmp --split 0
```

In the global configuration file the uniref30_mmseqs path needs to point to 
`your_directory_with_databases/uniref30_2202/uniref30_2202`

#### Species database

To use the standard Alphafold protocol for MSA pairing, GUIFold currently needs to create a database which maps accession to species identifiers.  You can download a precalculated accession-to-species-id database file based on uniprot (Nov 3rd 2021) from [https://www.dropbox.com/scl/fi/5gsrgoqhj74joo8ehlb4l/accession_species.db?rlkey=0uj0vnv96sv7bnuhi9ngjidmw&dl=0](https://www.dropbox.com/scl/fi/5gsrgoqhj74joo8ehlb4l/accession_species.db?rlkey=0uj0vnv96sv7bnuhi9ngjidmw&dl=0). The file needs to be placed into the uniprot folder (where the uniprot.fasta file is stored) and the filename needs to be "accession_species.db".

Alternatively you can create a new accession-to-species-id database from the uniprot.fasta file (this database will be also automatically created when the `colabfold_local` or `colabfold_web` protocols are used for the first time and the database file is not found in the expected location). If installation is done for a multi-user environment it is recommended to run this as part of the installation. 

1. Make sure the global configuraiton file is properly set up (esp. the uniprot database path needs to be defined)
2. Open the GUI (see Usage)
3. Paste any random sequence in FASTA format in the sequence input
4. Click `Read sequence` button
5. Select `colabfold_local` from the `Feature pipeline` dropdown menu
6. Click `Run` button
7. In the `Log` tab after some initialization, you should see the lines `Creating database...`. This step can take up to a few hours depending on hardware.

### (OPTIONAL) Configuration of cluster submission

**This step is only required if GUIFold is inteded to be used in combination with queuing systems (on a cluster).**

#### Settings

Queuing system specific settings need to be defined in the configuration file or in the `Settings`dialog of the GUI.

The following table shows configurations for common queueing systems:

Setting      | SLURM        | UGE/SGE
-----------: | -----------: | ----------:
Submit cmd   | sbatch       | qsub
Cancel cmd   | scancel      | qdel
JobID regex  | \D*(\d+)\D   | ?

Other settings:

`Min num CPUs` The minimum number of CPUs to request. This number of CPUs will be used for feature pipelines based on jackhmmer and hhblits. In our case a number of 20 CPUs works well.

`Max num CPUs` The maximum number of CPUs to request. This number is only used by feature pipelines based on mmseqs (currently only colabfold_local). In our case a number of 50 CPUs works well.

`Max RAM (GB)` The maximum RAM to request. This is only needed for certain tasks such as unified memory prediction or local mmseqs depending on the database (split) size (see setup of colabfold databases). We use 600 GB.

`Min RAM (GB)` The minimum RAM to request. This will be requested by default and should be high enough to avoid out of memory errors during the MSA step. In a few cases MSAs can get very large and require up to 120 GB.

`Max GPU memory (GB)` The maximum available GPU memory (on a single GPU model). 

`Split job` Whether to split the job into feature (CPU-only) and prediction steps (requires specific configuration of the submission script template, see below)

`Queue account` (Optional) Only needs to be set when an account is needed to submit jobs on a cluster. Each user has to define it individually.

`Use queue by default` Activates the `Submit to Queue` checkbox by default.

#### Preparation of a submission script template

The Jinja2 package is used to render the submission script template. See [Jinja2 documentation](https://jinja.palletsprojects.com/en/3.0.x/) for further information. The variables listed below can be used to create a template. An example file for a slurm cluster can be found in `guifold/templates/submission_script_slurm.j2`. The content of the file is also listed below ![link](example-of-a-template-for-SLURM).

The template file can be saved to any location. The path to the template needs to be set in the configuration file (see above) or in the `Settings` dialog of the GUI


#### Automatically splitting a job into CPU and GPU parts

If the queueing system supports dependencies (i.e. a job waits in the queue until another job has finished), the "split job feature" can be activated in the GUI settings if needed. Since the feature generation step does not require GPU, this step can be run on CPU-only resources. Two jobs will be submitted, the first job will request CPU (`use_gpu=False`) and the second job (`use_gpu=True`) will wait for the first job to finish (if the dependency is configured). An example how to add a dependency for SLURM and how to create a conditional to request CPU or GPU resources is provided below. Alternatively, the job can be manually devided into CPU and GPU steps by choosing `Only Features` in the GUI and, after this job has finished, re-starting the job with `Only Features` deactivated. 

#### Variables that can be used in the submission script template

GUIFold supports the following variables that can be used in the submission template. The parameters are determined based on the input and settings (configuration file or Settings dialog in the app):<br/><br/>
`{{logfile}}` (required) Path to the log file. Generated by the App.<br/>
`{{account}}` (optional) When a specific account is needed to run jobs on the cluster. As defined in settings.<br/>
`{{use_gpu}}` (optional) This can be used to build a conditional (example below) to select CPU or GPU nodes/queues<br/>
`{{mem}}` (optional) How much RAM (in GB) should be reserved. The RAM will be automatically increased with the GPU memory for unified memory. Depends on min/max RAM defined in settings and on the task.<br/>
`{{num_cpu}}` (optional) Number of CPUs to request (depends on the task and min/max CPUs defined in Settings)
`{{num_gpu}}` (optional) Number of GPUs to request (depends on the task and number of GPUs set in Advanced job settings)
`{{total_sequence_length}}` (optional) Total sequence length (not accounting for identical sequences). Useful to set runtime limits. Calculated by the App.<br/>
`{{gpu_mem}}` (optional) Useful when the queuing system supports selection of GPU by memory requirement. Calculated by the App (in GB unit) based on sequence length.<br/>
`{{split_mem}}` (optional) If the required memory exceeds the available GPU memory, the job can be run with unified memory. The split_mem variable holds None or the memory split fraction and can be used for a conditional to set the FLAGS required to enable unified memory use (see SLURM example below).<br/>
`{{add_dependency}}` (required) When the job is started with "split job setting", this variable will be True for the second job (prediction step) and allows adding a dependency on the first job (feature step). <br/>
`{{commnad}}` (required) The command to run the AlphaFold job. Generated by the App.

To cancel jobs from the GUI, the script also needs to write the Job ID to the logfile.
The pattern needs to be as follows:<br/>
```echo "QUEUE_JOB_ID=$JOB_ID_VARIABLE_FROM_QUEUING_SYSTEM"```<br/>
In case of SLURM it would be:<br/>
```echo "QUEUE_JOB_ID=$SLURM_JOB_ID"```

#### Example of a template for a SLURM cluster

The cluster in the example below has two types of GPUs, V100 (32 GB) and A100 (80 GB). The variable gpu_mem can be used
to build conditionals for choosing the appropriate GPU. 


```
#!/bin/bash
#SBATCH --account={{account}}
#SBATCH --job-name=alphafold
#SBATCH --cpus-per-task={{num_cpu}}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output={{logfile}}
#SBATCH --error={{logfile}}
#Append to logfile
#SBATCH --open-mode=append
#SBATCH --mem={{mem}}G


{% if use_gpu %}
#SBATCH --gres=gpu:1
{% if add_dependency %}
#If "Split Job" is selected in the GUI, add_dependency will be True for the second job and create a dependency on the first job (CPU-only)
#SBATCH --dependency=afterok:{{queue_job_id}}
#SBATCH --kill-on-invalid-dep=yes
{% endif %}
{% if gpu_mem|int <= 31 %}
#Select appropriate GPUs by e.g. constraint, nodename or gpu_name
#SBATCH --constraint=
#SBATCH --partition=
{% elif gpu_mem|int > 31 %}
#Select GPUs with > 31 GB memory by e.g. constraint, nodename or gpu_name
#SBATCH --constraint=
#SBATCH --partition=
{% endif %}

{% else %}
#If job only needs CPU
{% if total_sequence_length|int > 2000 %}
#SBATCH --partition=
{% else %}
#SBATCH --partition=
{% endif %}
{% endif %}


{% if split_mem %}
#If job needs to run with unified memory
export TF_FORCE_UNIFIED_MEMORY=True
export XLA_PYTHON_CLIENT_MEM_FRACTION={{split_mem}}
{% endif %}

echo "QUEUE_JOB_ID=$SLURM_JOB_ID"

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

When the conda env is activated (conda activate /path/to/af-conda) or added to PATH, LD_LIBRARY_PATH and PYTHONPATH (see [Setup of a module file](#setup-of-a-module-file) you can start GUIFold by typing:<br/>
`afgui.py`

To re-run an evaluation go to the job folder (where the FASTA sequence is stored) and type<br/>
`afeval.py --fasta_path name_of_sequence.fasta`

**See ![Wiki](https://github.com/fmi-basel/GUIFold/wiki/Usage) for more detailed documentation.**

## Licenses

GUIFold is licensed under the Apache License, Version 2.0.

Icons are from the GTK framework, licensed under [GPL](https://gitlab.gnome.org/GNOME/gtk/-/blob/main/COPYING).

The modified AlphaFold code retains its original license. See (https://github.com/deepmind/alphafold)

Third-party software and libraries may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## Citations

Some features were inspired by other projects and implemented from scratch if not indicated in the code.

[AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown)
[Colabfold](https://github.com/sokrypton/ColabFold)
[Alphafold](https://github.com/google-deepmind/alphafold)
