# VelMesh

VelMesh integrates two key tools developed for the CyberShake project:

- **Velocity Model Interpolator**: Supports 1D and 3D velocity models and generates input files compatible with AWP-ODC simulations (not for GPU at this moment).
- **Partition Tool**: Automatically computes optimal domain decompositions for parallel simulations.

Both tools are organized in independent folders but can be run **together (integrated)** or **independently**.

## Getting Started with the Repository

### Clone the Repository (Only for the First Time)
This command downloads a copy of the repository to your local machine.
```bash
git clone https://github.com/arianoesc27/VelMesh.git
```
### Navigate into the Repository Folder
Move into the project directory to start working with the repository.
```bash
cd VelMesh

```
### Update Your Local Repository with the Latest Changes
Fetch and merge the latest commits from the remote repository to stay up to date.
```bash
git pull origin main
```

## Example 3D velocity model

The 3D examples (`3DSocal_TestCase/` and `CyberShake_VelocityModelInterpolator/Example_3DVelocityModel/`)
use the SCEC CVM-S4.26 model in NetCDF format. The file is about 532 MB, too big to keep in this
repository, so it has to be downloaded once from SCEC. It is the file "CVM-S4.26 + 1D Background
Model NetCDF file" on this page.

https://strike.scec.org/scecpedia/SCEC_NetCDF_CVMS#CVM-S4.26

Save it with the name `cvms5_1d_depth_socal.nc` in the `inputData/` folder of each 3D example,
which is the name their `parameters.in` expect.
```bash
cp <downloaded file> 3DSocal_TestCase/inputData/cvms5_1d_depth_socal.nc
cp <downloaded file> CyberShake_VelocityModelInterpolator/Example_3DVelocityModel/inputData/cvms5_1d_depth_socal.nc
```

## Repository Structure

```
CyberShake_BSC_PartitionAndVelModel/
├── Cybershsake_VelocityModelInterpolator_BSC/   # Velocity model interpolation tool
├── Cybershsake_Partition/                       # Domain partitioning tool 
├── 3DSoCal_TestCase/                            # Integrated example case
```

---

## Instalación del entorno para usar el codigo en MN5

If you use the provided environment (environment.yml), dependencies will already be installed.
For local use, the script attempts to install the required packages automatically.
If that doesn't work, you can manually create and activate the environment on your local machine using the .yml file.
You can also use the progam from a shell using the included .sh script in the folder.

### MN5 (BSC)
The MN5 environment file and the CPU/GPU SLURM templates for MareNostrum 5 execution are available in the Running_MN5 folder.

```bash
# connect to glogin4 (or alogin4)
mamba env create -f environment_MN5.yml
mamba activate CyberShake-PartitionAndVelocityModel
```
---

## Recommended execution mode in MN5

The workflow is best run from an interactive session. Running interactively gives the user more control over the partitioning stage before launching the velocity-model interpolation. In particular, it allows you to:

- inspect the partitioning results,
- adjust partition-related parameters if needed,
- decide whether to continue to the interpolation/meshing stage,
- or rerun the partitioning step before proceeding.

### Example interactive session on MareNostrum 5

For CPU runs, an example interactive session is:

```bash
srun --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=112 --account=bsc21 --qos=gp_debug --pty bash
```

Once the interactive session is active, the workflow can be launched with:

```bash
./run_partition_and_velocity_model_GPU_batch.sh inputData/parameters.in
```
where the .sh file corresponds to the shell script provided in the Running_MN5 folder, either for the CPU or GPU workflow. 

### Batch execution with SLURM

The workflow can also be submitted directly through the provided SLURM scripts. In that case, the code will proceed automatically from the partitioning stage to the velocity-model generation stage, using the mesh and partitioning results obtained in the same run, without requiring user interaction.

---

## Integrated Execution: 3DSoCal_TestCase Example

The `3DSoCal_TestCase/` folder contains a full pipeline example that performs the following steps:

1. **Partition Phase**  
   Executes `run_Partition_and_velocity_model.sh` from the `3DSoCal_TestCase/` folder. After completing the partition, the code will update the `inputData/parameters.in` file with the computed dimensions and spacing.

2. **Velocity Model Interpolation Phase**  
   After the partitioning step, the interpolation is executed using the updated `parameters.in`.

3. **Results Storage**  
   All outputs from both steps are stored under the `outputData/` folder.

**Note:** 
- The values in the parameters.in file used in this example are for illustrative purposes and may need adjustment for different simulation scenarios.
- After the partitioning step, the code updates the parameters.in file in the inputData/ folder, as the new domain dimensions and spacing are required for the velocity model interpolation. The input parameters.in used during the partitioning process is saved in outputData/Partition/parameters_partition.in for reference. 

### How to Run the Integrated Pipeline

From the `3DSoCal_TestCase/` directory, run:

```bash
./run_partition_and_velocity_model_CPU.sh
```

This script ensures both phases are executed in sequence.

---

## Running Each Tool Independently

If you prefer to run either tool separately, refer to the `README.md` files and scripts within their respective folders:

### Velocity Model Interpolator  
Location: `Cybershsake_VelocityModelInterpolator_BSC/`
- Supports 1D and 3D models with or without attenuation
- Executable via Jupyter notebooks (`.ipynb`) or shell scripts (`.sh`)
- See the local `README.md` for full details

### Partition Tool  
Location: `Cybershsake_Partition/`
- Computes optimal domain decompositions based on `parameters.in`
- Executable using the `run_Partition.sh` script
- See the local `README.md` for full usage instructions

---

## Important Notes
- The `parameters.in` file is the core configuration and must be fully completed before running any tool.
- Ensure that the shell script have execution permissions (`chmod +x run_partition_and_velocity_model.sh`).

---

## License
This project is released under the BSD 3-Clause open-source license.

---
## Acknowledgments
This project was developed by Andrea C. Riaño (andrea.riano@bsc.es).  
Version 1.0.0, released in October 2026.
