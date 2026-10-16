# ucvm_to_netcdf

Writes a UCVM velocity model on a regular grid of latitude, longitude and depth, as a NetCDF file
with the layout the interpolator reads (`latitude`, `longitude`, `depth` in km, and `vp`, `vs` in
km/s and `rho` in kg/m3, dimensions in that order).

It exists to check the interpolator against UCVM. Both then start from the same model, so the
difference between the two meshes is only the error of going through a regular grid.

## Use

    python3 ucvm_to_netcdf.py --model cvmsi --ucvm-home <ucvm install> --output cvmsi_UCR_1km.nc \
        --model-coords model_coords_GC_UCR --margin-km 15 \
        --spacing-km 1.0 --depth-max-km 45 --depth-step-km 0.5 --jobs 8

`--model-coords` is the SCEC grid file of a site, and the box is that grid plus the margin. The
box can also be given with `--lon-min --lon-max --lat-min --lat-max`. Run it with the
interpolator's own Python (numpy and netCDF4). It needs a UCVM installation, and calls its
`ucvm_query` in several processes at once.

It also writes `<output>.json` with the model, the box, the spacing and the ranges of the values.

## Things to know

- The values are stored as UCVM gives them. No minimum Vs, Vp or density is applied. The
  interpolator and CyberShake's UCVM mesher each apply their own when they build the mesh.
- Depth is below the surface (UCVM geo-depth mode).
- The horizontal spacing is the parameter of the comparison. As fine as the mesh, the two meshes
  should agree. Coarser, the difference shows what is lost, mostly at basin edges.
- A 1 km grid of 329 x 378 x 91 points (11.3 million) took 38 s with 8 processes and is 71 MB.
  The time grows with the number of points, so 500 m is about 4 times that and 200 m about 25.
