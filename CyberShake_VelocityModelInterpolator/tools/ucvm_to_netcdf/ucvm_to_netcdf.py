#!/usr/bin/env python3
#AR: New (2026-10-06). Writes a UCVM velocity model as a NetCDF file the interpolator can read,
#AR: on a regular grid of latitude, longitude and depth.
#AR:
#AR: Why. To check the interpolator against UCVM both must start from the same model. With this
#AR: file the interpolator samples, say, CVM-S4.26.M01 (cvmsi) from NetCDF, UCVM samples the same
#AR: model directly on the same mesh, and what differs between the two meshes is the error of
#AR: going through a regular grid, which depends on how fine that grid is.
#AR:
#AR: How. The grid points are sent to UCVM's own ucvm_query, split in blocks of latitude rows
#AR: that run at the same time, and the answers are written as vp, vs and rho with the layout of
#AR: the models the interpolator already uses (dimensions latitude, longitude, depth).
#AR:
#AR: The values are stored as UCVM gives them, with NO minimum Vs, Vp or density applied. Both
#AR: the interpolator (VelocityModel3D) and CyberShake's UCVM mesher (ucvm-single_mpi) apply
#AR: their minimums when they build the mesh, so applying them here too would do it twice.
#AR:
#AR: Usage
#AR:   ucvm_to_netcdf.py --model cvmsi --ucvm-home <ucvm install> --output model.nc \
#AR:       --model-coords model_coords_GC_<SITE> --margin-km 15 \
#AR:       --spacing-km 1.0 --depth-max-km 45 --depth-step-km 0.5
#AR: or with the box given directly
#AR:       --lon-min -118.7 --lon-max -114.6 --lat-min 32.1 --lat-max 35.1
#AR: Needs numpy and netCDF4 (the interpolator's own Python) and a UCVM installation.

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

KM_PER_DEG = 111.19


def ucvm_env(ucvm_home):
    env = dict(os.environ)
    libs = ["lib/euclid3/lib", "lib/proj/lib", "lib/hdf5/lib", "lib/tiff/lib", "model/cvms/lib",
            "model/cvmsi/lib", "model/cvmh/lib", "model/sfcvm/lib"]
    env["LD_LIBRARY_PATH"] = ":".join([env.get("LD_LIBRARY_PATH", "")] +
                                      [os.path.join(ucvm_home, l) for l in libs])
    env["UCVM_INSTALL_PATH"] = ucvm_home
    env["PROJ_LIB"] = os.path.join(ucvm_home, "lib/proj/share/proj")
    #AR: Some model plugins start their own Python, so ours is not passed on.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def query_block(args):
    """Queries the rows lats of the grid. Returns (i0, array[len(lats), nlon, ndep, 3]) with
    vp, vs, rho in the units UCVM gives (m/s, m/s, kg/m3)."""
    i0, lats, lons, depths_m, model, ucvm_home = args
    n = len(lats) * len(lons) * len(depths_m)
    #AR: Longitude, latitude, depth in metres, one point per line, longitude changing fastest,
    #AR: then latitude, then depth. The reshape below relies on this order.
    #AR: (2026-10-08) Depth used to change fastest, so the same place was asked at one depth
    #AR: right after another. With cvmsi and cvms that changes the answer. Asked right after a
    #AR: depth between 0 and about 350 m at the same longitude and latitude, a point in that
    #AR: range comes back with another Vs (for example 960 m/s at 200 m after asking at 50 m,
    #AR: 1193 m/s asked alone, format_test/check_ucvm_state.py in the mesh validation folder).
    #AR: It was found when adding a level at 50 m changed 0.3 percent of the values at 200 m.
    #AR: With depth outermost no place is asked twice in a row, every value is the one UCVM
    #AR: gives for that point alone, and it is also the order of SCEC's mesher in its awp
    #AR: format, layer by layer.
    nlat, nlon, ndep = len(lats), len(lons), len(depths_m)
    dep_g = np.repeat(np.asarray(depths_m), nlat * nlon)
    lat_g = np.tile(np.repeat(np.asarray(lats), nlon), ndep)
    lon_g = np.tile(np.asarray(lons), nlat * ndep)
    text = "\n".join("%.6f %.6f %.2f" % t for t in zip(lon_g, lat_g, dep_g)) + "\n"
    cmd = [os.path.join(ucvm_home, "bin", "ucvm_query"), "-f", os.path.join(ucvm_home, "conf", "ucvm.conf"),
           "-m", model, "-c", "gd"]
    p = subprocess.run(cmd, input=text, capture_output=True, text=True, env=ucvm_env(ucvm_home))
    if p.returncode != 0:
        raise RuntimeError("ucvm_query failed (%d): %s" % (p.returncode, p.stderr[-500:]))
    #AR: One line of 17 columns per point, the last three are the combined vp, vs and rho.
    #AR: The message lines ucvm_query prints have fewer columns.
    rows = [r for r in (l.split() for l in p.stdout.splitlines()) if len(r) == 17]
    if len(rows) != n:
        raise RuntimeError("ucvm_query returned %d points, expected %d. Last error lines: %s"
                           % (len(rows), n, p.stderr[-300:]))
    out = np.array([[float(r[-3]), float(r[-2]), float(r[-1])] for r in rows], dtype=np.float32)
    #AR: Back to latitude, longitude, depth, the layout of the file.
    return i0, np.ascontiguousarray(out.reshape(ndep, nlat, nlon, 3).transpose(1, 2, 0, 3))


def main():
    ap = argparse.ArgumentParser(description="Writes a UCVM model on a regular lat/lon/depth grid as NetCDF.")
    ap.add_argument("--model", required=True, help="UCVM model name, for example cvmsi, cvms, cvmh")
    ap.add_argument("--ucvm-home", required=True, help="UCVM installation folder (the one with bin/ and conf/)")
    ap.add_argument("--output", required=True, help="NetCDF file to write")
    ap.add_argument("--model-coords", help="SCEC model_coords_GC_<SITE> file, the box is taken from it")
    ap.add_argument("--margin-km", type=float, default=15.0, help="margin around --model-coords, default 15")
    ap.add_argument("--lon-min", type=float)
    ap.add_argument("--lon-max", type=float)
    ap.add_argument("--lat-min", type=float)
    ap.add_argument("--lat-max", type=float)
    ap.add_argument("--spacing-km", type=float, help="horizontal spacing of the grid. Default, "
                    "the spacing of the mesh when --mesh-gridout is given")
    ap.add_argument("--depth-max-km", type=float)
    ap.add_argument("--depth-step-km", type=float)
    #AR: (2026-10-08) The depth levels of the simulation mesh itself, read from SCEC's gridout
    #AR: file, in place of --depth-max-km and --depth-step-km. The interpolator then finds the
    #AR: model at exactly the depths it asks for and does not interpolate in depth at all.
    #AR: --h-fraction adds the depth at which the surface layer is sampled, the same option of
    #AR: the mesher (0.25 gives a quarter of the spacing). One level is added below the deepest
    #AR: one so the bottom of the mesh is never at the very edge of the file.
    ap.add_argument("--mesh-gridout", help="SCEC gridout_<SITE> file, the depth levels are taken "
                    "from it instead of --depth-max-km and --depth-step-km")
    ap.add_argument("--h-fraction", type=float, default=0.0,
                    help="with --mesh-gridout, also adds the level at this fraction of the grid "
                         "spacing, where the mesher samples the surface layer. Default 0")
    #AR: (2026-10-08) Extra depth levels, on top of the regular ones. A mesher that samples the
    #AR: surface layer at a fraction of the grid spacing (--h_fraction 0.25 in SCEC's, 50 m for
    #AR: a 200 m grid) asks the model at a depth no regular level has. With that depth added
    #AR: here the interpolator reads UCVM's own value there, without it it draws a straight
    #AR: line from the level at 0 to the next one, which the model does not follow near the
    #AR: surface.
    ap.add_argument("--extra-depths-km", type=float, nargs="*", default=[],
                    help="extra depth levels in km, added to the regular ones, for example 0.05")
    ap.add_argument("--jobs", type=int, default=8, help="ucvm_query processes at the same time, default 8")
    a = ap.parse_args()

    if a.model_coords:
        c = np.loadtxt(a.model_coords, usecols=(0, 1))
        lat_mid = 0.5 * (c[:, 1].min() + c[:, 1].max())
        mlat = a.margin_km / KM_PER_DEG
        mlon = a.margin_km / (KM_PER_DEG * math.cos(math.radians(lat_mid)))
        lon_min, lon_max = c[:, 0].min() - mlon, c[:, 0].max() + mlon
        lat_min, lat_max = c[:, 1].min() - mlat, c[:, 1].max() + mlat
    elif None not in (a.lon_min, a.lon_max, a.lat_min, a.lat_max):
        lon_min, lon_max, lat_min, lat_max = a.lon_min, a.lon_max, a.lat_min, a.lat_max
    else:
        sys.exit("Give --model-coords, or the four of --lon-min --lon-max --lat-min --lat-max.")

    lat_mid = 0.5 * (lat_min + lat_max)
    dlat = a.spacing_km / KM_PER_DEG
    dlon = a.spacing_km / (KM_PER_DEG * math.cos(math.radians(lat_mid)))
    #AR: One more point than needed at the far end, so the grid always reaches the box.
    lats = lat_min + dlat * np.arange(int(math.ceil((lat_max - lat_min) / dlat)) + 1)
    lons = lon_min + dlon * np.arange(int(math.ceil((lon_max - lon_min) / dlon)) + 1)
    if a.mesh_gridout:
        with open(a.mesh_gridout) as f:
            g = f.readlines()
        nx_mb = int(g[1].split("=")[1])
        ny_mb = int(g[1 + nx_mb + 2].split("=")[1])
        zblock = 1 + nx_mb + 2 + ny_mb + 2
        nz = int(g[zblock].split("=")[1])
        mesh_z = np.array([float(g[zblock + 1 + k].split()[1]) for k in range(nz)])
        h = float(mesh_z[1] - mesh_z[0])
        depths_km = np.concatenate((mesh_z, [mesh_z[-1] + h]))
        if a.h_fraction > 0.0:
            depths_km = np.concatenate((depths_km, [a.h_fraction * h]))
        depths_km = np.unique(np.round(depths_km, 6))
        if a.spacing_km is None:
            a.spacing_km = h
        a.depth_step_km = h
        print("Depth levels of the mesh in %s, %d levels every %.3f km%s" % (
            a.mesh_gridout, nz, h, (", plus one at %.4f km" % (a.h_fraction * h)) if a.h_fraction > 0 else ""))
    elif a.depth_max_km is None or a.depth_step_km is None or a.spacing_km is None:
        sys.exit("Give --mesh-gridout, or the three of --spacing-km --depth-max-km --depth-step-km.")
    else:
        depths_km = a.depth_step_km * np.arange(int(math.ceil(a.depth_max_km / a.depth_step_km)) + 1)
    if a.extra_depths_km:
        if min(a.extra_depths_km) < 0.0 or max(a.extra_depths_km) > depths_km[-1]:
            sys.exit("--extra-depths-km must be between 0 and %.3f km." % depths_km[-1])
        depths_km = np.unique(np.round(np.concatenate((depths_km, a.extra_depths_km)), 6))
        print("Extra depth levels at %s km" % ", ".join("%g" % d for d in sorted(a.extra_depths_km)))
    npts = len(lats) * len(lons) * len(depths_km)
    print("Model %s, box lon %.4f..%.4f lat %.4f..%.4f, depth 0..%.2f km" %
          (a.model, lons[0], lons[-1], lats[0], lats[-1], depths_km[-1]))
    print("Grid %d x %d x %d = %d points, spacing %.3f km (%.5f deg lat, %.5f deg lon), every %.3f km in depth"
          % (len(lats), len(lons), len(depths_km), npts, a.spacing_km, dlat, dlon, a.depth_step_km))
    print("About %.1f GB before compression" % (npts * 3 * 4 / 1e9), flush=True)

    #AR: Blocks of about 250 thousand points, so no ucvm_query gets more than a few tens of MB.
    rows_per_block = max(1, int(250000 // (len(lons) * len(depths_km))))
    tasks = [(i, lats[i:i + rows_per_block], lons, depths_km * 1000.0, a.model, a.ucvm_home)
             for i in range(0, len(lats), rows_per_block)]

    import netCDF4
    tmp = a.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    nc = netCDF4.Dataset(tmp, "w", format="NETCDF4")
    nc.createDimension("latitude", len(lats))
    nc.createDimension("longitude", len(lons))
    nc.createDimension("depth", len(depths_km))
    for name, vals, units in (("latitude", lats, "degrees_north"), ("longitude", lons, "degrees_east"),
                              ("depth", depths_km, "km")):
        v = nc.createVariable(name, "f8", (name,))
        v[:] = vals
        v.units = units
    out = {}
    for name, units, long_name in (("vp", "km.s-1", "P wave velocity"), ("vs", "km.s-1", "S wave velocity"),
                                   ("rho", "kg.m-3", "density")):
        #AR: (2026-10-07) One chunk per latitude row, the same piece the loop below writes at a
        #AR: time. With the library's own choice of chunks every row written touched chunks that
        #AR: held other rows too, so each of them was read back, uncompressed, changed and
        #AR: compressed again. At 200 m that was 7 s per row in this process alone, whatever
        #AR: the number of ucvm_query running (measured with 32 and with 104, same speed).
        out[name] = nc.createVariable(name, "f4", ("latitude", "longitude", "depth"), zlib=True, complevel=1,
                                      chunksizes=(1, len(lons), len(depths_km)))
        out[name].units = units
        out[name].long_name = long_name
    nc.title = "UCVM model %s on a regular grid" % a.model
    nc.source = "ucvm_query -m %s -c gd, values as given by UCVM, no minimum velocity or density applied" % a.model
    nc.query_order = "one depth level after another, no place asked twice in a row"
    nc.ucvm_model = a.model
    nc.horizontal_spacing_km = a.spacing_km
    nc.depth_step_km = a.depth_step_km
    if a.extra_depths_km:
        nc.extra_depths_km = ", ".join("%g" % d for d in sorted(a.extra_depths_km))
    nc.depth_reference = "depth below the surface, UCVM geo-depth mode"
    nc.history = "%s ucvm_to_netcdf.py %s" % (datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), " ".join(sys.argv[1:]))

    t0 = time.time()
    done, blocks, bad = 0, 0, 0
    lo = {"vp": 1e30, "vs": 1e30, "rho": 1e30}
    hi = {"vp": -1e30, "vs": -1e30, "rho": -1e30}
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i0, block in pool.map(query_block, tasks):
            n = block.shape[0]
            #AR: UCVM gives m/s, the file is in km/s like the other models the interpolator reads.
            for k, name in enumerate(("vp", "vs", "rho")):
                vals = block[..., k] / (1000.0 if name != "rho" else 1.0)
                out[name][i0:i0 + n] = vals
                lo[name] = min(lo[name], float(vals.min()))
                hi[name] = max(hi[name], float(vals.max()))
            bad += int(np.sum(block[..., 1] <= 0.0))
            done += n
            blocks += 1
            if done == len(lats) or blocks % 20 == 0:
                print("  %d of %d latitude rows, %.0f s" % (done, len(lats), time.time() - t0), flush=True)
    nc.close()
    os.replace(tmp, a.output)

    meta = dict(model=a.model, output=os.path.abspath(a.output), lon=[float(lons[0]), float(lons[-1])],
                lat=[float(lats[0]), float(lats[-1])], depth_km=[0.0, float(depths_km[-1])],
                shape=[len(lats), len(lons), len(depths_km)], spacing_km=a.spacing_km,
                depth_step_km=a.depth_step_km, extra_depths_km=sorted(a.extra_depths_km),
                minimums_applied=False, points_without_data=bad,
                ranges={k: [lo[k], hi[k]] for k in lo}, seconds=round(time.time() - t0, 1),
                command=" ".join(sys.argv))
    with open(os.path.splitext(a.output)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Written %s in %.0f s" % (a.output, time.time() - t0))
    for k in ("vp", "vs", "rho"):
        print("  %-3s %.4g to %.4g %s" % (k, lo[k], hi[k], "km/s" if k != "rho" else "kg/m3"))
    if bad:
        print("WARNING, %d points have no data in this model (vs <= 0). The interpolator will refuse a "
              "domain that touches them. Use a smaller box or a model that covers it." % bad)


if __name__ == "__main__":
    main()
