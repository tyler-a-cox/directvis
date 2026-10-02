# directvis

Direct-summation visibility simulator for a handful of point sources. It
takes the same arguments as `fftvis.simulate_vis` and returns visibilities in
the same layout. On top of that:

- source fluxes, including full Stokes, can **change from one integration to the next**;
- large outputs can be **written in time chunks** to a memory map, HDF5 or zarr
  array, or streamed with `simulate_vis_chunks`.

For each source, the beam is interpolated only along that source's track and
each visibility is summed directly. No NUFFT is involved:

```
V_ij[p, q](f, t) = [A_j^H C(f, t) A_i]_qp · exp(2πi f (x_j − x_i)·s(t) / c)
```

Here `A` is the beam Jones matrix and `C` the source coherency in the local alt/az frame.

## When to use it

The cost scales as N_sources × N_baselines × N_times × N_freqs. For one or a
few sources that is far less work than an FFT-based simulator, whose transform
costs about the same however many sources there are. For full skies, use fftvis
or matvis.

Example: one polarized source with time-varying flux, HERA-350 (6611 unique
baselines), 1536 channels, 400 times, a tabulated efield UVBeam. On the same
2-core machine:

| | Wall time |
|---|---|
| fftvis, one call per time step | ~5 h (projected from single-call timings) |
| directvis, one call | 175 s measured, 1.6 GB peak memory |

That run produces 260 GB of complex128 output, so in practice it is written
out in chunks (see below).

## Install

Requires Python ≥ 3.11.

```bash
git clone <repo-url> directvis
cd directvis
pip install .            # or: pip install -e .  (editable)
pip install ".[test]"    # adds pytest, fftvis, pyradiosky and pyuvsim for the tests
```

Dependencies: numpy, scipy, astropy, pyuvdata ≥ 3.1.2 and matvis ≥ 1.3.2.
fftvis is **not** required; it is only used by the tests.

## Usage

The call is the same as fftvis:

```python
import numpy as np
import directvis

vis = directvis.simulate_vis(
    ants=ants,                  # {antenna: [e, n, u]} in meters
    fluxes=fluxes,              # see the shapes below
    ra=ra, dec=dec,             # radians, shape (Nsrc,)
    freqs=freqs,                # Hz
    times=times,                # Julian dates or astropy Time
    beam=uvbeam,                # UVBeam, AnalyticBeam, BeamInterface, or a list + beam_idx
    telescope_loc=hera_loc,     # astropy EarthLocation
    polarized=True,
)
# polarized=True  -> (Nfreqs, Ntimes, 2, 2, Nbls)
# polarized=False -> (Nfreqs, Ntimes, Nbls)
```

`baselines` defaults to one baseline per redundant group, including the
autocorrelation, in the same order fftvis uses. fftvis-only options such as
`eps`, `nprocesses` or `upsample_factor` are accepted and ignored, with a
warning, so existing calls run unchanged.

### Flux shapes

| `fluxes` shape | Meaning |
|---|---|
| `(Nsrc, Nfreqs)` | Stokes I, constant in time |
| `(Nsrc, Nfreqs, 4)` | Stokes I, Q, U, V, constant in time (`polarized=True` only) |
| `(Nsrc, Ntimes, Nfreqs)` | Stokes I per time |
| `(Nsrc, Ntimes, Nfreqs, 4)` | Stokes I, Q, U, V per time (`polarized=True` only) |

The first two shapes mean exactly what they mean in fftvis. Stokes I is split
equally between the two linear feeds, as in fftvis.

### Large outputs

Write straight to disk, one block of times at a time:

```python
out = np.lib.format.open_memmap("vis.npy", mode="w+", dtype=complex,
                                shape=(len(freqs), len(times), 2, 2, len(baselines)))
directvis.simulate_vis(..., baselines=baselines, out=out)
```

Anything that supports `out[:, i:j] = chunk` works as `out`, for example an
h5py dataset or a zarr array. Or handle each block yourself:

```python
for tslice, chunk in directvis.simulate_vis_chunks(..., times_per_chunk=8):
    save(tslice, chunk)   # the buffer is reused: write or copy it before the next one
```

## Conventions and validation

The following all follow fftvis:

- coordinates (matvis `CoordinateRotation`, ERFA by default);
- the coherency definition and its rotation into the alt/az frame;
- UVBeam interpolation and the Jones vector-component ordering;
- the output layout.

The test suite checks directvis against fftvis to its NUFFT accuracy (about 1e-13) for:

- polarized and unpolarized modes, Stokes I and full-Stokes skies;
- constant and time-varying fluxes, the latter against one fftvis call per time;
- several sources, including one below the horizon;
- flat and non-flat arrays;
- UVBeam and analytic beams.

Baselines between antennas with **different beams** are checked against
pyuvsim instead. fftvis 1.1.2 and earlier compute those wrongly when
`polarized=True`.

## Tests

```bash
pip install ".[test]"
pytest
```

The pyuvsim comparison runs only when `mpi4py` is importable. Install it with
`pip install mpi4py`, plus an MPI library: your system's, or `pip install mpich`.

## License

MIT
