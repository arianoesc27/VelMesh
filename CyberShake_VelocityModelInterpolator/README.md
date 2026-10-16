# Cybershsake_VelocityModelInterpolator

The provided scripts allow users to generate files compatible with **AWP-ODC**, the SCEC software for Anelastic Wave Propagation - Olsen, Day, Cui (AWP-ODC) using the format compatible with Cybershake scripts (RWG). The scripts in this repository can be run using either Jupyter notebooks or shell scripts for automation.

## Software Structure

```
├── Example_1DVelocityModel_Elastic/
│   ├── inputData/   # Contains the 1D velocity model (CSV or TXT format)
│   ├── outputData/  # Generated files after running the code
│
├── Example_1DVelocityModel_QpQs/
│   ├── inputData/   # 1D velocity model with attenuation (CSV or TXT)
│   ├── outputData/  # Generated output files
│
├── Example_3DVelocityModel/
│   ├── inputData/   # 3D velocity model (CSV, TXT, NC, JSON)
│   ├── outputData/  # Generated files after running the code
│
├── VelocityModel_utils/ # Utility functions for handling velocity models
│
├── A_Interpolate_1D_Model.ipynb  # Jupyter notebook for 1D model interpolation
├── B_Interpolate_3D_Model.ipynb  # Jupyter notebook for 3D model interpolation
│
├── run_1Dinterpolator.sh  # Shell script for automatic execution of 1D models
├── run_3Dinterpolator.sh  # Shell script for automatic execution of 3D models
```

## Running the Examples

### Using Jupyter Notebooks - standalone execution
The provided notebooks **A_Interpolate_1D_Model.ipynb** and **B_Interpolate_3D_Model.ipynb** guide users through the interpolation process step by step. These are recommended for understanding how the interpolation works before using the automated shell scripts.

### Using Shell Scripts - standalone execution
For automated execution, you can use the provided `.sh` scripts:

```bash
./run_1Dinterpolator.sh  # Runs the 1D interpolation with input files
./run_3Dinterpolator.sh  # Runs the 3D interpolation with input files
```

These scripts generate the **outputData/** folder, which contains:
- `AWP_inputfiles/` → Coordinates files and binary model for AWP-ODC
- `Figures/` → Maps and plots of the interpolated velocity model
- `parameters.in` → Copy of the configuration file used to define the model and stored in inputData/
- `input_vm.csv (or .txt, .nc, or .json)` → Copy of the input velocity model used in the run stored in inputData/ 
- `metadata.json` → Metadata containing details of the run

## Recommended Workflow
1. **Run the example notebooks** to understand how the input files are structured and how the velocity model is processed.
2. **Use the shell scripts** to automate interpolation using the VM input file and the configuration defined in `parameters.in`.
3. **Check the output data** to verify correct interpolation.

## Input File Formats
### Depth units 
Depth values should be provided in **kilometers (km)**.  

### Velocity and Density Units  
- **Vp (P-wave velocity)**: Can be provided in **km/s** or **m/s**. 
- **Vs (S-wave velocity)**: Can be provided in **km/s** or **m/s**. 
- **Rho (Density)**: Can be provided in **g/cm³** or **kg/m³**. 

### Supported File Formats  
- **1D Models**: Accepts **CSV** or **TXT** files containing depth-dependent velocity properties.  
- **3D Models**: Accepts **CSV**, **TXT**, **NetCDF (`.nc`)**, or **JSON** formats for volumetric velocity models.  
- All variable names must be in **lowercase** (e.g., `depth`, `latitude`, `longitude`, `vp`, `vs`, `rho`) 

## Notes
- Running the example notebooks first is recommended before moving to automated execution.
- When using `.sh` scripts, ensure the necessary input files are provided in the `inputData/` folder (parameters.in and the VM input file).
- `parameters.in` is only used by the shell scripts and not by the Jupyter notebooks. Make sure to complete all the variables in the parameter file. The `parameters.in` file provided in the examples is configured to produce the same results as the Jupyter notebooks, except for the grip dynamic maps, which are not printed.
- It is recommended to delete the `outputData/` folder before re-executing a run.
- Large binary files can take a while to be created (a 5GB can take up to 40 min to finish). Interpolation functions were oprtimized to run batches in parallel. 

## License
This project is released under the BSD 3-Clause open-source license.

## Acknowledgment
This is an open-source project developed by Andrea C. Riaño (andrea.riano@bsc.es).
Version 1.0.0, released in October 2026.

## Minimum velocities of the mesh

Every mesh point gets these corrections, in this order, the same SCEC's UCVM mesher applies
(`ucvm-single_mpi.c` in cybershake-core).

1. Vs below the minimum is raised to it, and Vp is raised by the same factor, so the point keeps
   its Vp/Vs.
2. Vp below its minimum is raised to it.
3. Density below its minimum is raised to it.
4. Vp/Vs is never left below 1.45. If it is, Vs is lowered to Vp/1.45. This is a physical limit
   and does not change with the region.

One number sets the three minimums, the minimum Vs. The rule is the one of SCEC's
`UCVM/single_exe.csh`.

| Minimum Vs (m/s) | Minimum Vp (m/s) | Minimum density (kg/m3) |
|---|---|---|
| 500, the default | 1700 | 1700 |
| any other value | 3.4 times the minimum Vs | the same number as the minimum Vp |
| 400, for example | 1360 | 1360 |
| 900, a special case in SCEC's script | 1800 | 2000 |

It is given with `--min-vs` to `Interp_velocity_3D_awp.py`, or with `input.model.minVs` in the
workflow's JSON, which sends the same value to the UCVM route. Left out, the mesh is built with
500 m/s, as it always was.

Two things to know. The density minimum equal to the Vp minimum as a number is what SCEC's
script does, and it is followed here so that the two meshers agree, not because of a physical
reason known to us. And lowering the minimum Vs shortens the shortest wavelength, so the grid
spacing has to go down with it to keep the points per wavelength, minimum Vs / (maximum
frequency x spacing). CyberShake uses 5, which is 500 m/s at 0.5 Hz and 200 m. With 400 m/s at
0.5 Hz that needs 160 m. The workflow's preparation phase prints this number and warns below 5.

## Choosing them in the standalone 3D program

`Interp_velocity_3D.py` reads three optional keys from `parameters.in`. Without them the mesh
is the one the program has always written.

| Key | Default | What it does |
|---|---|---|
| `min_vs` | 500 | Minimum Vs in m/s. The minimum Vp and density follow it with the rule above. |
| `format` | `awp` | `awp` or `awpz`, the rule for Vp at surface points raised to the minimum Vs. |
| `h_fraction` | 0.25 | Depth of the surface layer as a fraction of the grid spacing. |

The standalone program has always sampled the surface a quarter of the grid spacing down, which
is why its default is 0.25, while `Interp_velocity_3D_awp.py` defaults to 0 as the SCEC mesher
does. The 1D program does not read these keys.
