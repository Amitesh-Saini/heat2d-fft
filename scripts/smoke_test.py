from heat2d_io import load_run, colormap_for, provenance_label

run = load_run("output/data/heat_gaussian.h5")   # no ../
print(run.nt, run.nx, run.ny, run.Lx, run.alpha, run.ic_type)
print(colormap_for(run))
print(provenance_label(run))
print(list(run.diagnostics.keys()))
print(run.for_plotting(0).shape)