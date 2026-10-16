"""
###########################################################################
#        Velocity model -> AWP media on SCEC's grid (CyberShake mode)      #
#--------------------------------------------------------------------------
# CyberShake mode of VelMesh. Used by the CyberShake workflow when        #
# input.model.source is "interpolator". Samples a 3D velocity model on the #
# grid SCEC's own gen_model_cords produced, and writes the AWP media file  #
# in the same layout UCVM writes it. See                                   #
# write_awp_media_from_scec_grid() in VelocityModel_utils.py for why.      #
###########################################################################
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import VelocityModel_utils  # noqa: E402

__version__ = "1.0.0"


# Unit spellings seen in model files, mapped to the names used in the JSON config.
_UNIT_ALIASES = {
    "km/s": "km/s", "km.s-1": "km/s", "km s-1": "km/s", "kilometers per second": "km/s",
    "m/s": "m/s", "m.s-1": "m/s", "m s-1": "m/s", "meters per second": "m/s",
    "kg/m3": "kg/m3", "kg.m-3": "kg/m3", "kg m-3": "kg/m3", "kg/m^3": "kg/m3",
    "g/cm3": "g/cm3", "g.cm-3": "g/cm3", "g cm-3": "g/cm3", "g/cm^3": "g/cm3",
}


def units_in_file(path):
    """Units the model file itself declares, as {'vp': ..., 'vs': ..., 'rho': ...}, or {} if it
    declares none or is not a NetCDF file."""
    if not path.lower().endswith((".nc", ".nc4", ".netcdf")):
        return {}
    try:
        import netCDF4
    except ImportError:
        return {}
    out = {}
    with netCDF4.Dataset(path) as d:
        for v in ("vp", "vs", "rho"):
            if v in d.variables and hasattr(d.variables[v], "units"):
                raw = str(d.variables[v].units).strip().lower()
                out[v] = _UNIT_ALIASES.get(raw, raw)
    return out


def check_units(path, velocity_units, density_units):
    """#AR: The units stay in the JSON on purpose, a model file may not declare them. But when it
    #AR: does and they disagree, stop. On 2026-09-17 a JSON saying g/cm3 for a NetCDF whose rho was
    #AR: already kg/m3 wrote every density 1000 times too large, and nothing complained."""
    declarados = units_in_file(path)
    if not declarados:
        print("Units: %s does not declare any, using the config's (velocity %s, density %s)"
              % (os.path.basename(path), velocity_units, density_units))
        return
    esperado = {"vp": velocity_units, "vs": velocity_units, "rho": density_units}
    malos = ["  %s: file says %s, config says %s" % (k, declarados[k], esperado[k])
             for k in ("vp", "vs", "rho") if k in declarados and declarados[k] != esperado[k]]
    if malos:
        raise SystemExit(
            "ERROR: the units in the config do not match the ones %s declares:\n%s\nFix "
            "velocityUnits / densityUnits in the JSON." % (path, "\n".join(malos)))
    print("Units: config matches the file (velocity %s, density %s)" % (velocity_units, density_units))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1].strip("# "))
    ap.add_argument("--vm-file", required=True, help="velocity model file (NetCDF)")
    ap.add_argument("--velocity-units", required=True, choices=["km/s", "m/s"])
    ap.add_argument("--density-units", required=True, choices=["g/cm3", "kg/m3"])
    ap.add_argument("--model-coords", required=True, help="SCEC model_coords_GC_<site>")
    ap.add_argument("--gridout", required=True, help="SCEC gridout_<site>")
    #AR: (2026-10-06) One number sets the three minimums of the mesh, with SCEC's rule
    #AR: (VelocityModel_utils.scec_minimums). Left out, the mesh is built as before.
    ap.add_argument("--min-vs", type=float, default=500.0,
                    help="minimum Vs of the mesh in m/s, default 500. The minimum Vp is 3.4 times "
                         "it and the minimum density that same number, as in SCEC's UCVM mesher")
    #AR: (2026-10-08) The two options of SCEC's mesher, same names and defaults, see
    #AR: write_awp_media_from_scec_grid in VelocityModel_utils.py.
    ap.add_argument("--format", default="awp", choices=VelocityModel_utils.SURFACE_FORMATS,
                    help="awp, the default, or awpz. With awpz a surface point raised to the "
                         "minimum Vs takes the Vp/Vs ratio of the point below it, 4 at most, "
                         "which avoids too large a Vp. Same as --format of SCEC's single_exe.py")
    ap.add_argument("--h-fraction", type=float, default=0.0,
                    help="depth at which the surface layer is sampled, as a fraction of the grid "
                         "spacing. Default 0, the surface itself. SCEC's Study 24.8 used 0.25. "
                         "Same as --h_fraction of SCEC's single_exe.py")
    ap.add_argument("--output", required=True, help="output media file, e.g. awp.<site>.media")
    args = ap.parse_args()

    print("Interp_velocity_3D_awp.py version %s" % __version__)
    check_units(args.vm_file, args.velocity_units, args.density_units)

    loader = VelocityModel_utils.VelocityModelLoader(args.vm_file)
    model_data = loader.load()

    # No-data check on the raw values, before any unit conversion.
    coords = np.loadtxt(args.model_coords, usecols=(0, 1), dtype=np.float64)
    with open(args.gridout) as f:
        g = f.readlines()
    nx_mb = int(g[1].split("=")[1])
    ny_mb = int(g[1 + nx_mb + 2].split("=")[1])
    zblock = 1 + nx_mb + 2 + ny_mb + 2
    nz = int(g[zblock].split("=")[1])
    max_depth = float(g[zblock + nz].split()[1])
    VelocityModel_utils.check_model_has_no_fill_values(model_data, coords[:, 1], coords[:, 0],
                                                       max_depth)
    del coords

    model_data = VelocityModel_utils.convert_units(model_data, args.velocity_units,
                                                   args.density_units)
    velocity_model = VelocityModel_utils.VelocityModel3D(model_data, min_vs=args.min_vs)
    print("Minimums applied to the mesh: Vs %.1f m/s, Vp %.1f m/s, density %.1f kg/m3, Vp/Vs %.2f"
          % (velocity_model.min_vs, velocity_model.min_vp, velocity_model.min_rho,
             VelocityModel_utils.MIN_VP_VS_RATIO))
    print("Format %s, surface layer at %.2f of the grid spacing" % (args.format, args.h_fraction))
    VelocityModel_utils.write_awp_media_from_scec_grid(args.model_coords, args.gridout,
                                                       velocity_model, args.output,
                                                       fmt=args.format, h_fraction=args.h_fraction)
    size = os.path.getsize(args.output)
    expected = nx_mb * ny_mb * nz * 3 * 4
    if size != expected:
        raise SystemExit("ERROR: %s is %d bytes, expected %d" % (args.output, size, expected))
    print("--> Computation completed! %s (%d bytes)" % (args.output, size))


if __name__ == "__main__":
    main()
