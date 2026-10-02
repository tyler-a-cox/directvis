"""Direct-summation visibility simulator for a few point sources.

``simulate_vis`` takes the same arguments as ``fftvis.simulate_vis`` and returns
visibilities in the same layout, with two additions:

* ``fluxes`` may carry a time axis, so source flux and polarization can vary
  from one integration to the next;
* the output can be written in time chunks to any array-like (``out=``), or
  streamed with :func:`simulate_vis_chunks`.

Each source contributes, for feeds (p, q),

    V_ij[p, q](f, t) = [A_j^H C(f, t) A_i]_{qp} * exp(2 pi i f (x_j - x_i) . s(t) / c)

where A is the beam Jones matrix (sky vector component x feed) and C the source
coherency in the local alt/az frame. It is evaluated directly: the beam is
interpolated only along each source's track and no NUFFT is involved. Cost
scales as Nsrc x Nbls x Ntimes x Nfreqs, so this is the right tool for a handful
of sources; use fftvis or matvis for full skies.

Coordinates (matvis ``CoordinateRotation``), the coherency definition and its
equatorial-to-alt/az rotation, UVBeam interpolation, the Jones vector-component
ordering and the output layout all follow fftvis, and the results agree with
fftvis to its NUFFT accuracy. Baselines between antennas with different beams
match pyuvsim; fftvis 1.1.2 and earlier get those wrong for polarized
simulations.

Example
-------
A polarized source whose Stokes parameters vary with time, written to disk one
block of times at a time::

    import directvis
    import numpy as np

    # stokes: (Ntimes, Nfreqs, 4) array of [I, Q, U, V] in Jy
    out = np.lib.format.open_memmap("vis.npy", mode="w+", dtype=complex,
                                    shape=(len(freqs), len(times), 2, 2, len(baselines)))
    directvis.simulate_vis(ants=ants, fluxes=stokes[None], ra=np.array([ra]), dec=np.array([dec]),
                           freqs=freqs, times=times, beam=uvbeam, telescope_loc=hera_loc,
                           polarized=True, baselines=baselines, out=out)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from matvis.coordinates import calc_coherency_rotation, enu_to_az_za
from matvis.core.beams import prepare_beam_unpolarized
from matvis.core.coords import CoordinateRotation
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AnalyticBeam
from pyuvdata.beam_interface import BeamInterface
from scipy.interpolate import interp1d

import matvis.cpu.coords  # noqa: F401  (registers the CPU coordinate-rotation classes)

__all__ = ["simulate_vis", "simulate_vis_chunks", "get_pos_reds"]

SPEED_OF_LIGHT = 299792458.0  # m/s

# fftvis keywords that only control its NUFFT or parallelization; accepted and ignored
_IGNORED_FFTVIS_KWARGS = {
    "eps", "upsample_factor", "flat_array_tol", "nprocesses", "nthreads",
    "force_use_type3", "force_use_ray", "trace_mem", "backend", "max_memory",
    "min_chunks", "source_buffer",
}

# target size of one time chunk of output, in bytes
_CHUNK_BYTES = 256 * 1024**2


def get_pos_reds(antpos: dict, decimals: int = 3, include_autos: bool = True) -> list[list[tuple]]:
    """Group baselines into redundant sets from antenna positions.

    Same grouping and ordering as ``fftvis.utils.get_pos_reds``, so the default
    baseline list matches fftvis's.

    Parameters
    ----------
    antpos : dict
        ``{antenna: np.array([e, n, u])}`` in meters.
    decimals : int
        Baseline vectors are rounded to this many decimals (meters) before grouping.
    include_autos : bool
        Whether to include the autocorrelation group.

    Returns
    -------
    list of lists of (ant1, ant2)
        Redundant groups; within each group the first baseline has a non-negative
        north component.
    """
    uv_to_red_key = {}
    reds = {}
    pairs = [(ai, aj) for ai in antpos for aj in antpos if ai < aj or (include_autos and ai == aj)]
    blvecs = np.round([antpos[aj] - antpos[ai] for ai, aj in pairs], decimals)
    for (ai, aj), (u, v, _) in zip(pairs, blvecs):
        if (u, v) not in uv_to_red_key and (-u, -v) not in uv_to_red_key:
            reds[(ai, aj)] = [(ai, aj)]
            uv_to_red_key[(u, v)] = (ai, aj)
        elif (-u, -v) in uv_to_red_key:
            reds[uv_to_red_key[(-u, -v)]].append((aj, ai))
        else:
            reds[uv_to_red_key[(u, v)]].append((ai, aj))

    reds_list = []
    for red in reds.values():
        ant1, ant2 = red[0]
        if (antpos[ant2] - antpos[ant1])[1] < 0:
            reds_list.append([(b, a) for a, b in red])
        else:
            reds_list.append(red)
    return reds_list


def simulate_vis(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray | Time,
    beam: UVBeam | AnalyticBeam | BeamInterface | Sequence,
    telescope_loc: EarthLocation,
    beam_idx: np.ndarray | None = None,
    baselines: list[tuple] | None = None,
    precision: int = 2,
    polarized: bool = False,
    beam_spline_opts: dict | None = None,
    use_feed: Literal["x", "y"] = "x",
    interpolation_function: str = "az_za_map_coordinates",
    coord_method: Literal["CoordinateRotationAstropy", "CoordinateRotationERFA"] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    out=None,
    times_per_chunk: int | None = None,
    **fftvis_kwargs,
):
    """Simulate visibilities for a few point sources by direct summation.

    Parameters
    ----------
    ants : dict
        Antenna positions ``{antenna: [e, n, u]}`` in meters (ENU).
    fluxes : np.ndarray
        Source flux density in Jy. Accepted shapes:

        - ``(Nsrc, Nfreqs)``: Stokes I, constant in time;
        - ``(Nsrc, Nfreqs, 4)``: Stokes I, Q, U, V, constant in time (``polarized=True`` only);
        - ``(Nsrc, Ntimes, Nfreqs)``: Stokes I per time;
        - ``(Nsrc, Ntimes, Nfreqs, 4)``: Stokes I, Q, U, V per time (``polarized=True`` only).

        As in fftvis, Stokes I is split equally between the two linear feeds,
        so an unpolarized source of flux S gives S / 2 in each of xx and yy.
        A 3-D array that fits both ``(Nsrc, Nfreqs, 4)`` and
        ``(Nsrc, Ntimes, Nfreqs)`` is read the fftvis way, as constant Stokes.
    ra, dec : np.ndarray
        ICRS source positions in radians, shape ``(Nsrc,)``.
    freqs : np.ndarray
        Frequencies in Hz.
    times : np.ndarray or astropy.time.Time
        Observation times (Julian dates if an array).
    beam : UVBeam, AnalyticBeam, BeamInterface, or a list of them
        Antenna beam(s). With a list, ``beam_idx`` maps antennas to beams; it may
        be omitted when the list has one beam or one beam per antenna.
    telescope_loc : astropy.coordinates.EarthLocation
        Array center.
    beam_idx : np.ndarray, optional
        Beam index for each antenna, in the order of ``ants``.
    baselines : list of (ant1, ant2), optional
        Baselines to simulate. Defaults to one baseline per redundant group,
        including the autocorrelation, as in fftvis.
    precision : {1, 2}
        Output dtype: 1 for complex64, 2 for complex128. Internal arithmetic is
        always double precision.
    polarized : bool
        If True, return all four feed products and use the beam's full Jones
        matrix (an efield beam is required). If False, return the ``use_feed``
        product using the corresponding power beam.
    beam_spline_opts : dict, optional
        ``spline_opts`` passed to :meth:`pyuvdata.UVBeam.interp`.
    use_feed : {"x", "y"}
        Feed used when ``polarized=False``.
    interpolation_function : str
        UVBeam spatial interpolation function.
    coord_method : str
        matvis coordinate rotation class used for source positions.
    coord_method_params : dict, optional
        Extra arguments for the coordinate rotation class.
    out : array-like, optional
        Destination for the result, with the shape and dtype described under
        Returns. Anything that supports ``out[:, i:j] = chunk`` works, e.g. a
        NumPy array, ``np.memmap``, h5py dataset or zarr array. If omitted, an
        in-memory array is allocated.
    times_per_chunk : int, optional
        Number of times computed per step. Defaults to roughly 256 MB of output
        per chunk.
    **fftvis_kwargs
        fftvis NUFFT and parallelization options (``eps``, ``nprocesses``, ...)
        are accepted so calls can be switched between packages, and ignored.

    Returns
    -------
    vis : array-like
        ``(Nfreqs, Ntimes, 2, 2, Nbls)`` if ``polarized``, else
        ``(Nfreqs, Ntimes, Nbls)``. This is ``out`` when it is given.

    Raises
    ------
    ValueError
        For inconsistent input shapes, a full-Stokes sky with ``polarized=False``,
        or a ``beam_idx`` that does not match the beams and antennas.
    TypeError
        For keyword arguments that neither package accepts.
    NotImplementedError
        For fftvis eigenbeam inputs (``beam_coefs``).
    """
    sim = _Simulation(
        ants, fluxes, ra, dec, freqs, times, beam, telescope_loc, beam_idx, baselines,
        precision, polarized, beam_spline_opts, use_feed, interpolation_function,
        coord_method, coord_method_params, fftvis_kwargs,
    )
    if out is None:
        out = np.empty(sim.shape, dtype=sim.dtype)
    elif tuple(out.shape) != sim.shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {sim.shape}")

    write_in_place = isinstance(out, np.ndarray) and out.dtype == sim.dtype
    buf = None
    for tsl in sim.time_chunks(times_per_chunk):
        if write_in_place:
            sim.compute_chunk(tsl, out=out[:, tsl])
        else:
            buf = sim.compute_chunk(tsl, out=_buffer_for(buf, sim, tsl))
            out[:, tsl] = buf
    return out


def simulate_vis_chunks(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray | Time,
    beam: UVBeam | AnalyticBeam | BeamInterface | Sequence,
    telescope_loc: EarthLocation,
    beam_idx: np.ndarray | None = None,
    baselines: list[tuple] | None = None,
    precision: int = 2,
    polarized: bool = False,
    beam_spline_opts: dict | None = None,
    use_feed: Literal["x", "y"] = "x",
    interpolation_function: str = "az_za_map_coordinates",
    coord_method: Literal["CoordinateRotationAstropy", "CoordinateRotationERFA"] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    times_per_chunk: int | None = None,
    **fftvis_kwargs,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield the simulation one block of times at a time.

    Takes the same arguments as :func:`simulate_vis` (except ``out``) and yields
    ``(time_slice, vis_chunk)``, where ``vis_chunk`` covers ``times[time_slice]``
    in the :func:`simulate_vis` layout. The chunk buffer is reused, so write or
    copy each chunk before requesting the next.
    """
    sim = _Simulation(
        ants, fluxes, ra, dec, freqs, times, beam, telescope_loc, beam_idx, baselines,
        precision, polarized, beam_spline_opts, use_feed, interpolation_function,
        coord_method, coord_method_params, fftvis_kwargs,
    )
    buf = None
    for tsl in sim.time_chunks(times_per_chunk):
        buf = sim.compute_chunk(tsl, out=_buffer_for(buf, sim, tsl))
        yield tsl, buf


def _buffer_for(buf, sim, tsl):
    n = tsl.stop - tsl.start
    shape = (sim.nfreqs, n) + sim.shape[2:]
    if buf is None or buf.shape != shape:
        buf = np.empty(shape, dtype=sim.dtype)
    return buf


class _Simulation:
    """Everything that is computed once per call; visibilities are then produced per time chunk."""

    def __init__(
        self, ants, fluxes, ra, dec, freqs, times, beam, telescope_loc, beam_idx=None, baselines=None,
        precision=2, polarized=False, beam_spline_opts=None, use_feed="x",
        interpolation_function="az_za_map_coordinates", coord_method="CoordinateRotationERFA",
        coord_method_params=None, fftvis_kwargs=None,
    ):
        _check_fftvis_kwargs(fftvis_kwargs or {})
        if precision not in (1, 2):
            raise ValueError("precision must be 1 or 2")

        self.polarized = bool(polarized)
        self.dtype = np.complex64 if precision == 1 else np.complex128
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        self.times = times if isinstance(times, Time) else Time(np.asarray(times), format="jd")
        if self.times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
        dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
        if ra.shape != dec.shape or ra.ndim != 1:
            raise ValueError("ra and dec must be 1-D arrays of the same length")
        self.nsrc, self.nfreqs, self.ntimes = ra.size, self.freqs.size, len(self.times)

        self.flux, self.sky_polarized, self.flux_varies = _parse_fluxes(
            fluxes, self.nsrc, self.ntimes, self.nfreqs, self.polarized
        )

        antnums = list(ants.keys())
        antpos = {a: np.asarray(ants[a], dtype=np.float64) for a in antnums}
        if baselines is None:
            baselines = [red[0] for red in get_pos_reds(antpos, include_autos=True)]
        self.baselines = [tuple(b) for b in baselines]
        unknown = {a for bl in self.baselines for a in bl} - set(antnums)
        if unknown:
            raise ValueError(f"baselines refer to antennas not in ants: {sorted(unknown)}")
        self.nbls = len(self.baselines)

        beam_list = list(beam) if isinstance(beam, (list, tuple)) else [beam]
        beam_idx = _validate_beam_idx(beam_idx, len(beam_list), len(antnums))
        ant_beam = dict(zip(antnums, beam_idx))

        # Baselines grouped by the (beam of ant1, beam of ant2) pair they need
        self.pair_bls: dict[tuple[int, int], np.ndarray] = {}
        for k, (a1, a2) in enumerate(self.baselines):
            self.pair_bls.setdefault((ant_beam[a1], ant_beam[a2]), []).append(k)
        self.pair_bls = {p: np.asarray(v) for p, v in self.pair_bls.items()}
        used_beams = sorted({b for p in self.pair_bls for b in p})

        # Source directions, ENU unit vectors, shape (3, Ntimes, Nsrc)
        self.topo = _source_track(ra, dec, self.times, telescope_loc, coord_method, coord_method_params)
        self.up = self.topo[2] > 0  # (Ntimes, Nsrc); sources below the horizon contribute nothing

        # Beam response along each source track, for every beam that is used
        az, za = enu_to_az_za(enu_e=self.topo[0][self.up], enu_n=self.topo[1][self.up], orientation="uvbeam")
        self.beam_resp = {}
        for b in used_beams:
            resp = _beam_response(
                beam_list[b], az, za, self.freqs, self.polarized, use_feed,
                interpolation_function, beam_spline_opts,
            )
            full = np.zeros(resp.shape[:-1] + self.up.shape, dtype=resp.dtype)
            full[..., self.up] = resp
            self.beam_resp[b] = full  # (2, 2, Nf, Nt, Nsrc) if polarized else (Nf, Nt, Nsrc)

        if self.polarized and self.sky_polarized:
            self.coherency = _coherency_altaz(self.flux, ra, dec, self.topo, self.up, self.times, telescope_loc)

        # Geometric delays (x_j - x_i) . s / c, shape (Nsrc, Ntimes, Nbls)
        blvec = np.array([antpos[a2] - antpos[a1] for a1, a2 in self.baselines])
        self.tau = np.einsum("bk,kts->stb", blvec, self.topo) / SPEED_OF_LIGHT

    @property
    def shape(self):
        if self.polarized:
            return (self.nfreqs, self.ntimes, 2, 2, self.nbls)
        return (self.nfreqs, self.ntimes, self.nbls)

    def time_chunks(self, times_per_chunk=None):
        if times_per_chunk is None:
            per_time = self.nfreqs * self.nbls * (4 if self.polarized else 1) * np.dtype(np.complex128).itemsize
            times_per_chunk = max(1, int(_CHUNK_BYTES // per_time))
        for t0 in range(0, self.ntimes, times_per_chunk):
            yield slice(t0, min(t0 + times_per_chunk, self.ntimes))

    def _apparent(self, pair, tsl):
        """Beam-weighted coherency for one beam pair, (Nsrc, Nf, Ntc, 2, 2) or (Nsrc, Nf, Ntc)."""
        bi, bj = pair
        if not self.polarized:
            # power beams: sqrt(P_i P_j) * I / 2
            p = np.sqrt(self.beam_resp[bi][:, tsl] * self.beam_resp[bj][:, tsl])  # (Nf, Ntc, Nsrc)
            flux = self.flux[:, tsl if self.flux_varies else slice(None)]  # (Nsrc, Nt|1, Nf)
            return 0.5 * np.moveaxis(p, -1, 0) * np.swapaxes(flux, 1, 2)

        # Jones vector components in fftvis order (UVBeam's vector axis reversed)
        Ai = np.flip(self.beam_resp[bi][:, :, :, tsl], axis=0)  # (ax, feed, Nf, Ntc, Nsrc)
        Aj = np.flip(self.beam_resp[bj][:, :, :, tsl], axis=0)
        if self.sky_polarized:
            C = self.coherency[:, tsl]  # (Nsrc, Ntc, Nf, 2, 2), alt/az frame
            out = np.einsum("apfts,stfac,cqfts->sftpq", Aj.conj(), C, Ai, optimize=True)
        else:
            flux = self.flux[:, tsl if self.flux_varies else slice(None)]  # (Nsrc, Nt|1, Nf)
            out = 0.5 * np.einsum("apfts,aqfts->sftpq", Aj.conj(), Ai, optimize=True)
            out *= np.swapaxes(flux, 1, 2)[..., None, None]
        # feed product (p, q) is element (q, p) of A_j^H C A_i
        return np.swapaxes(out, -1, -2)

    def compute_chunk(self, tsl, out):
        """Write visibilities for times[tsl] into ``out`` (Nf, Ntc, ...) and return it."""
        ntc = tsl.stop - tsl.start
        app = {pair: self._apparent(pair, tsl) for pair in self.pair_bls}
        comps = [(p, q) for p in range(2) for q in range(2)] if self.polarized else [()]
        phase = np.empty((self.nfreqs, ntc, self.nbls), dtype=np.complex128)
        scratch = {}
        written = set()
        for s in range(self.nsrc):
            if not self.up[tsl, s].any():
                continue
            tau = self.tau[s, tsl]  # (Ntc, Nbls)
            for fi, f in enumerate(self.freqs):
                arg = (2 * np.pi * f) * tau
                np.cos(arg, out=phase[fi].real)
                np.sin(arg, out=phase[fi].imag)
            for pair, idx in self.pair_bls.items():
                every_bl = len(idx) == self.nbls
                ph = phase if every_bl else phase[..., idx]
                for comp in comps:
                    a = app[pair][(s, Ellipsis) + comp][..., None]  # (Nf, Ntc, 1)
                    key = (slice(None), slice(None)) + comp
                    if every_bl and pair not in written:
                        np.multiply(a, ph, out=out[key])
                        continue
                    tmp = scratch.get(ph.shape)
                    if tmp is None:
                        tmp = scratch[ph.shape] = np.empty(ph.shape, dtype=np.complex128)
                    np.multiply(a, ph, out=tmp)
                    if not every_bl:
                        key = key + (idx,)
                    if pair in written:
                        out[key] += tmp
                    else:
                        out[key] = tmp
                written.add(pair)
        for pair, idx in self.pair_bls.items():
            if pair not in written:
                out[..., idx] = 0
        return out


def _check_fftvis_kwargs(kwargs):
    if "beam_coefs" in kwargs and kwargs["beam_coefs"] is not None:
        raise NotImplementedError("eigenbeam inputs (beam_coefs) are not supported")
    unknown = set(kwargs) - _IGNORED_FFTVIS_KWARGS - {"beam_coefs"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    ignored = sorted(k for k in kwargs if k in _IGNORED_FFTVIS_KWARGS)
    if ignored:
        warnings.warn(f"directvis ignores the fftvis NUFFT/parallelization options {ignored}", stacklevel=4)


def _parse_fluxes(fluxes, nsrc, ntimes, nfreqs, polarized):
    """Return (flux, sky_polarized, varies_in_time).

    flux is (Nsrc, Nt or 1, Nf) for Stokes I, or (Nsrc, Nt or 1, Nf, 4) for full Stokes.
    """
    fluxes = np.asarray(fluxes)
    if np.iscomplexobj(fluxes):
        raise ValueError("fluxes must be real")
    fluxes = fluxes.astype(np.float64, copy=False)
    shp = fluxes.shape
    stokes_const = (nsrc, nfreqs, 4)
    i_varying = (nsrc, ntimes, nfreqs)

    if fluxes.ndim == 2 and shp == (nsrc, nfreqs):
        return fluxes[:, None, :], False, False
    if fluxes.ndim == 3:
        if shp == stokes_const and polarized:
            if shp == i_varying:
                warnings.warn(
                    f"fluxes shape {shp} fits both (Nsrc, Nfreqs, 4) and (Nsrc, Ntimes, Nfreqs); "
                    "reading it as constant Stokes. Pass (Nsrc, Ntimes, Nfreqs, 4) for time-variable flux.",
                    stacklevel=4,
                )
            return fluxes[:, None], True, False
        if shp == i_varying:
            return fluxes, False, True
        if shp == stokes_const:
            raise ValueError("full-Stokes fluxes (Nsrc, Nfreqs, 4) require polarized=True")
    if fluxes.ndim == 4 and shp == (nsrc, ntimes, nfreqs, 4):
        if not polarized:
            raise ValueError("full-Stokes fluxes (Nsrc, Ntimes, Nfreqs, 4) require polarized=True")
        return fluxes, True, True
    raise ValueError(
        f"fluxes has shape {shp}; expected (Nsrc, Nfreqs), (Nsrc, Nfreqs, 4), (Nsrc, Ntimes, Nfreqs) "
        f"or (Nsrc, Ntimes, Nfreqs, 4) with Nsrc={nsrc}, Ntimes={ntimes}, Nfreqs={nfreqs}"
    )


def _validate_beam_idx(beam_idx, nbeam, nant):
    if beam_idx is None:
        if nbeam == 1:
            return np.zeros(nant, dtype=int)
        if nbeam == nant:
            return np.arange(nant)
        raise ValueError(f"beam_idx is required when there are {nbeam} beams for {nant} antennas")
    beam_idx = np.asarray(beam_idx)
    if beam_idx.shape != (nant,):
        raise ValueError(f"beam_idx must have length {nant} (one entry per antenna)")
    if not np.issubdtype(beam_idx.dtype, np.integer):
        raise ValueError("beam_idx must contain integers")
    if beam_idx.min() < 0 or beam_idx.max() >= nbeam:
        raise ValueError(f"beam_idx entries must be in [0, {nbeam})")
    return beam_idx


def _source_track(ra, dec, times, telescope_loc, coord_method, coord_method_params):
    """ENU unit vectors of every source at every time, shape (3, Ntimes, Nsrc)."""
    mgr = CoordinateRotation._methods[coord_method](
        flux=np.ones((ra.size, 1)),
        times=times,
        telescope_loc=telescope_loc,
        skycoords=SkyCoord(ra=ra * un.rad, dec=dec * un.rad, frame="icrs"),
        precision=2,
        **(coord_method_params or {}),
    )
    mgr.setup()
    topo = np.empty((3, len(times), ra.size))
    for ti in range(len(times)):
        mgr.rotate(ti)
        topo[:, ti] = mgr.all_coords_topo
    return topo


def _coherency_altaz(stokes, ra, dec, topo, up, times, telescope_loc):
    """Source coherency in the local alt/az frame, shape (Nsrc, Ntimes, Nf, 2, 2)."""
    I, Q, U, V = np.moveaxis(stokes, -1, 0)  # each (Nsrc, Nt|1, Nf)
    coh = 0.5 * np.array([[I + Q, U + 1j * V], [U - 1j * V, I - Q]])  # (2, 2, Nsrc, Nt|1, Nf)
    coh = np.moveaxis(coh, (0, 1), (-2, -1))  # (Nsrc, Nt|1, Nf, 2, 2)
    nsrc, ntimes = up.shape[1], up.shape[0]
    out = np.zeros((nsrc, ntimes) + coh.shape[2:], dtype=np.complex128)
    az, za = enu_to_az_za(enu_e=topo[0], enu_n=topo[1], orientation="astropy")  # (Nt, Nsrc)
    for ti in range(ntimes):
        s = np.flatnonzero(up[ti])
        if s.size == 0:
            continue
        R = calc_coherency_rotation(
            ra=ra[s], dec=dec[s], alt=np.pi / 2 - za[ti, s], az=az[ti, s], time=times[ti], location=telescope_loc
        )  # (2, 2, ns)
        c = coh[s, ti if coh.shape[1] > 1 else 0]  # (ns, Nf, 2, 2)
        out[s, ti] = np.einsum("ban,nfbc,cdn->nfad", R, c, R)
    return out


def _beam_response(beam, az, za, freqs, polarized, use_feed, interpolation_function, spline_opts):
    """Beam response at the given directions.

    Returns (Naxes_vec, Nfeeds, Nf, Npts) Jones values if ``polarized``, else
    (Nf, Npts) power-beam values for ``use_feed``.
    """
    bi = beam if isinstance(beam, BeamInterface) else BeamInterface(beam)
    kw = dict(interpolation_function=interpolation_function, spline_opts=spline_opts,
              reuse_spline=True, check_azza_domain=False)

    if polarized:
        if bi.beam_type != "efield":
            raise ValueError("polarized=True requires efield beams")
        if not bi._isuvbeam:
            return bi.compute_response(az_array=az, za_array=za, freq_array=freqs)
        return _uvbeam_jones(bi.beam, az, za, freqs, kw)

    if not bi._isuvbeam:
        return prepare_beam_unpolarized(bi, use_feed=use_feed).compute_response(
            az_array=az, za_array=za, freq_array=freqs
        )[0, 0].real
    return _uvbeam_power(bi.beam, az, za, freqs, use_feed, kw)


def _freq_lookup(beam_freqs, freqs, tol=1.0):
    """Indices of exact (within ``tol`` Hz) beam channels for every frequency, or None."""
    dist = np.abs(beam_freqs[None, :] - freqs[:, None])
    if np.all(dist.min(axis=1) < tol):
        return dist.argmin(axis=1)
    return None


def _uvbeam_jones(uvb, az, za, freqs, kw):
    """Efield UVBeam along the track: spatial interpolation at the beam's own channels, then in frequency.

    Both steps are linear in the beam data and act on different axes, so this
    equals interpolating the full beam in frequency first (as fftvis does).
    """
    data, _ = uvb.interp(az_array=az, za_array=za, freq_array=None, return_basis_vector=False, **kw)
    bf = np.asarray(uvb.freq_array).ravel()
    nearest = _freq_lookup(bf, freqs)
    if nearest is not None:
        return data[:, :, nearest]
    if bf.size == 1:
        raise ValueError("the UVBeam has a single frequency, which does not match the simulated frequencies")
    if freqs.min() < bf.min() or freqs.max() > bf.max():
        raise ValueError(f"simulated frequencies extend outside the UVBeam's range {[bf.min(), bf.max()]}")
    re = interp1d(bf, data.real, kind="cubic", axis=2)(freqs)
    im = interp1d(bf, data.imag, kind="cubic", axis=2)(freqs)
    return re + 1j * im


def _uvbeam_power(uvb, az, za, freqs, use_feed, kw):
    """Power beam for ``use_feed`` along the track.

    Follows fftvis's order: interpolate the efield beam in frequency, convert to
    power, then interpolate spatially. Frequencies are processed in blocks to
    bound memory.
    """
    nbytes_per_chan = uvb.data_array.nbytes / max(uvb.Nfreqs, 1)
    block = max(1, int(_CHUNK_BYTES // max(nbytes_per_chan, 1)))
    out = np.empty((freqs.size, az.size))
    for f0 in range(0, freqs.size, block):
        fb = freqs[f0:f0 + block]
        b = uvb.interp(freq_array=fb, new_object=True, run_check=False) if uvb.Nfreqs > 1 else uvb
        pb = prepare_beam_unpolarized(BeamInterface(b), use_feed=use_feed)
        out[f0:f0 + block] = pb.compute_response(az_array=az, za_array=za, freq_array=fb, **kw)[0, 0].real
    return out
