"""Tests for directvis.

Comparisons against fftvis are skipped if fftvis is not installed, and the pyuvsim
comparison is skipped without pyuvsim and mpi4py.
"""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata.analytic_beam import AiryBeam, ShortDipoleBeam

import directvis

LOC = EarthLocation.from_geodetic(lat=-30.72152612068957, lon=21.428303826863015, height=1051.69)
FREQS = np.array([112.3e6, 137.9e6, 151.0e6, 188.8e6])  # between the test beam's channels
NTIMES = 5
TIMES = 2459845.25 + np.arange(NTIMES) * 120.0 / 86400.0
REL_TOL = 1e-9  # fftvis default NUFFT accuracy is 1e-13; leave room for sums over sources


def hex_array(hex_num=3, sep=14.6):
    pos = []
    for row in range(hex_num - 1, -hex_num, -1):
        for col in range(2 * hex_num - abs(row) - 1):
            pos.append([sep * ((2 - (2 * hex_num - abs(row))) / 2 + col), row * sep * np.sqrt(3) / 2, 0.0])
    return dict(enumerate(np.array(pos)))


def make_uvbeam(width_deg=20.0, scale=1.0):
    bf = np.arange(100e6, 201e6, 5e6)
    az = np.deg2rad(np.arange(0, 360, 2.0))
    za = np.deg2rad(np.arange(0, 91, 2.0))
    uvb = ShortDipoleBeam().to_uvbeam(freq_array=bf, beam_type="efield", axis1_array=az, axis2_array=za)
    fwhm = np.deg2rad(width_deg) * (150e6 / bf)
    taper = np.exp(-0.5 * (za[None, :] / (fwhm[:, None] / 2.355)) ** 2)
    uvb.data_array = uvb.data_array * taper[None, None, :, :, None] * scale
    return uvb


def sky():
    """Near-zenith source, an off-zenith source, and one that never rises at HERA."""
    lst = Time(TIMES[NTIMES // 2], format="jd", location=LOC).sidereal_time("apparent").rad
    ra = np.array([lst, lst + 0.25, lst])
    dec = np.array([LOC.lat.rad + 0.03, LOC.lat.rad - 0.3, np.deg2rad(65.0)])
    return ra, dec


def stokes(nsrc, ntimes=None, pol=True):
    rng = np.random.default_rng(1)
    shape = (nsrc, NTIMES, FREQS.size) if ntimes else (nsrc, FREQS.size)
    I = 5 + rng.random(shape)
    if not pol:
        return I
    frac = rng.uniform(-0.5, 0.5, shape + (3,)) / np.sqrt(3)
    return np.concatenate([I[..., None], I[..., None] * frac], axis=-1)


def rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture(scope="module")
def ants_flat():
    return hex_array()


@pytest.fixture(scope="module")
def ants_rough():
    rng = np.random.default_rng(0)
    return {a: p + rng.normal(0, [0.3, 0.3, 0.5]) for a, p in hex_array().items()}


@pytest.fixture(scope="module")
def uvbeam():
    return make_uvbeam()


try:
    import fftvis
except ImportError:  # pragma: no cover
    fftvis = None
needs_fftvis = pytest.mark.skipif(fftvis is None, reason="fftvis not installed")


def run_both(ants, fluxes, beam, polarized, per_time=False, **kw):
    ra, dec = sky()
    dv = directvis.simulate_vis(ants=ants, fluxes=fluxes, ra=ra, dec=dec, freqs=FREQS, times=TIMES,
                                beam=beam, telescope_loc=LOC, polarized=polarized, **kw)
    if per_time:
        fv = np.concatenate([
            fftvis.simulate_vis(ants=ants, fluxes=fluxes[:, t], ra=ra, dec=dec, freqs=FREQS,
                                times=TIMES[t:t + 1], beam=beam, telescope_loc=LOC, polarized=polarized, **kw)
            for t in range(NTIMES)
        ], axis=1)
    else:
        fv = fftvis.simulate_vis(ants=ants, fluxes=fluxes, ra=ra, dec=dec, freqs=FREQS, times=TIMES,
                                 beam=beam, telescope_loc=LOC, polarized=polarized, **kw)
    return dv, fv


@needs_fftvis
@pytest.mark.parametrize("layout", ["ants_flat", "ants_rough"])
def test_polarized_sky_constant(layout, uvbeam, request):
    dv, fv = run_both(request.getfixturevalue(layout), stokes(3), uvbeam, polarized=True)
    assert dv.shape == fv.shape
    assert rel_err(dv, fv) < REL_TOL


@needs_fftvis
def test_polarized_sky_time_variable(ants_rough, uvbeam):
    dv, fv = run_both(ants_rough, stokes(3, ntimes=True), uvbeam, polarized=True, per_time=True)
    assert rel_err(dv, fv) < REL_TOL


@needs_fftvis
@pytest.mark.parametrize("ntimes", [None, True])
def test_polarized_beam_unpolarized_sky(ants_rough, uvbeam, ntimes):
    dv, fv = run_both(ants_rough, stokes(3, ntimes, pol=False), uvbeam, polarized=True, per_time=bool(ntimes))
    assert rel_err(dv, fv) < REL_TOL


@needs_fftvis
@pytest.mark.parametrize("use_feed", ["x", "y"])
@pytest.mark.parametrize("ntimes", [None, True])
def test_unpolarized(ants_rough, uvbeam, use_feed, ntimes):
    dv, fv = run_both(ants_rough, stokes(3, ntimes, pol=False), uvbeam, polarized=False,
                      per_time=bool(ntimes), use_feed=use_feed)
    assert dv.shape == fv.shape == (FREQS.size, NTIMES, dv.shape[-1])
    assert rel_err(dv, fv) < REL_TOL


@needs_fftvis
def test_analytic_beams(ants_flat):
    dv, fv = run_both(ants_flat, stokes(3), ShortDipoleBeam(), polarized=True)
    assert rel_err(dv, fv) < REL_TOL
    dv, fv = run_both(ants_flat, stokes(3, pol=False), AiryBeam(diameter=14.0), polarized=False)
    assert rel_err(dv, fv) < REL_TOL


@needs_fftvis
def test_default_baselines_match_fftvis(ants_rough):
    ref = [red[0] for red in fftvis.utils.get_pos_reds(ants_rough, include_autos=True)]
    assert [red[0] for red in directvis.get_pos_reds(ants_rough, include_autos=True)] == ref


def both_orders(ants):
    keys = list(ants)
    return [(a, b) for a in keys for b in keys if a != b] + [(keys[0], keys[0])]


def complex_beam_pair():
    """Two efield UVBeams whose Jones matrices differ by a complex, angle-dependent factor."""
    b0 = make_uvbeam(20.0)
    b1 = make_uvbeam(28.0, scale=0.9)
    za = b1.axis2_array
    b1.data_array[:, 1] *= np.exp(1j * za)[None, None, :, None]
    b1.data_array[0] *= 0.8
    return [b0, b1]


@needs_fftvis
def test_multiple_beams_unpolarized(ants_rough):
    beams = complex_beam_pair()
    beam_idx = np.array([a % 2 for a in ants_rough])
    dv, fv = run_both(ants_rough, stokes(3, pol=False), beams, polarized=False,
                      beam_idx=beam_idx, baselines=both_orders(ants_rough))
    assert rel_err(dv, fv) < REL_TOL


@pytest.mark.parametrize("polarized_sky", [False, True])
def test_multiple_beams_vs_pyuvsim(polarized_sky):
    """Different, complex beams on the two antennas, both baseline orientations, against pyuvsim."""
    pytest.importorskip("mpi4py")
    pyuvsim = pytest.importorskip("pyuvsim")
    from astropy import units as un
    from astropy.coordinates import Latitude, Longitude
    from matvis._test_utils import get_standard_sim_params
    from pyradiosky import SkyModel
    from pyuvdata import BeamInterface
    from pyuvsim import simsetup
    from pyuvsim.telescope import BeamList

    params, _, uvbeams, _, uvdata = get_standard_sim_params(use_analytic_beam=False, polarized=True, nants=3, ntime=3)
    ants = params.pop("ants")
    b0 = uvbeams.beam_list[0].beam
    b1 = b0.copy()
    b1.data_array[:, 1] *= np.exp(0.6j)
    b1.data_array[0] *= 0.8
    beam_idx = np.array([0, 1, 0])
    bls = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2)]

    rng = np.random.default_rng(3)
    nsrc = len(params["ra"])
    st = np.zeros((4, 1, nsrc))
    st[0] = rng.uniform(1, 2, (1, nsrc))
    if polarized_sky:
        st[1:] = st[0] * rng.uniform(-0.3, 0.3, (3, 1, nsrc))
    sky_model = SkyModel(name=[str(i) for i in range(nsrc)], ra=Longitude(params["ra"], unit="rad"),
                         dec=Latitude(params["dec"], unit="rad"), spectral_type="flat", stokes=st * un.Jy, frame="icrs")
    uvd = pyuvsim.uvsim.run_uvdata_uvsim(
        uvdata, BeamList([BeamInterface(b0, beam_type="efield"), BeamInterface(b1, beam_type="efield")]),
        beam_dict={str(a): int(b) for a, b in zip(ants, beam_idx)}, catalog=simsetup.SkyModelData(sky_model),
    )
    ref = np.array([[[uvd.get_data(bl + (pol,))[:, 0] for pol in row] for row in (("xx", "xy"), ("yx", "yy"))]
                    for bl in bls])  # (Nbls, 2, 2, Ntimes)

    dv = directvis.simulate_vis(
        ants=ants, fluxes=np.transpose(st, (2, 1, 0)) if polarized_sky else st[0].T, ra=params["ra"],
        dec=params["dec"], freqs=params["freqs"], times=params["times"], beam=[b0, b1], beam_idx=beam_idx,
        baselines=bls, telescope_loc=params["telescope_loc"], polarized=True,
        coord_method="CoordinateRotationAstropy", interpolation_function="az_za_simple",
    )
    np.testing.assert_allclose(np.transpose(dv[0], (3, 1, 2, 0)), ref, atol=1e-8 * np.abs(ref).max())


def test_hermitian_symmetry_multiple_beams(ants_rough):
    beams = complex_beam_pair()
    beam_idx = np.array([a % 2 for a in ants_rough])
    bls = both_orders(ants_rough)
    ra, dec = sky()
    v = directvis.simulate_vis(ants=ants_rough, fluxes=stokes(3), ra=ra, dec=dec, freqs=FREQS, times=TIMES,
                               beam=beams, beam_idx=beam_idx, baselines=bls, telescope_loc=LOC, polarized=True)
    k = {bl: i for i, bl in enumerate(bls)}
    for (a, b), i in k.items():
        j = k.get((b, a))
        if j is not None:
            np.testing.assert_allclose(v[..., j], np.swapaxes(v[..., i], 2, 3).conj(), atol=1e-12)


def test_out_and_chunks_match(ants_rough, uvbeam, tmp_path):
    ra, dec = sky()
    kw = dict(ants=ants_rough, fluxes=stokes(3, ntimes=True), ra=ra, dec=dec, freqs=FREQS, times=TIMES,
              beam=uvbeam, telescope_loc=LOC, polarized=True)
    full = directvis.simulate_vis(**kw)

    mm = np.lib.format.open_memmap(tmp_path / "vis.npy", mode="w+", dtype=full.dtype, shape=full.shape)
    assert directvis.simulate_vis(**kw, out=mm, times_per_chunk=2) is mm
    np.testing.assert_array_equal(np.asarray(mm), full)

    class ListBacked:  # minimal non-NumPy array-like, written through __setitem__
        def __init__(self, shape):
            self.shape, self.data = shape, np.zeros(shape, complex)

        def __setitem__(self, key, value):
            self.data[key] = value

    lb = directvis.simulate_vis(**kw, out=ListBacked(full.shape), times_per_chunk=3)
    np.testing.assert_array_equal(lb.data, full)

    seen = np.zeros(NTIMES, bool)
    for tsl, chunk in directvis.simulate_vis_chunks(**kw, times_per_chunk=2):
        np.testing.assert_array_equal(chunk, full[:, tsl])
        seen[tsl] = True
    assert seen.all()


def test_precision_1(ants_flat, uvbeam):
    ra, dec = sky()
    kw = dict(ants=ants_flat, fluxes=stokes(3), ra=ra, dec=dec, freqs=FREQS, times=TIMES,
              beam=uvbeam, telescope_loc=LOC, polarized=True)
    v1 = directvis.simulate_vis(**kw, precision=1)
    v2 = directvis.simulate_vis(**kw, precision=2)
    assert v1.dtype == np.complex64 and v2.dtype == np.complex128
    assert rel_err(v1, v2) < 1e-6


def test_below_horizon_source_gives_zero(ants_flat, uvbeam):
    ra, dec = sky()
    v = directvis.simulate_vis(ants=ants_flat, fluxes=stokes(1), ra=ra[2:], dec=dec[2:], freqs=FREQS,
                               times=TIMES, beam=uvbeam, telescope_loc=LOC, polarized=True)
    assert not np.any(v)


def test_flux_shape_checks(ants_flat, uvbeam):
    ra, dec = sky()
    kw = dict(ants=ants_flat, ra=ra, dec=dec, freqs=FREQS, times=TIMES, beam=uvbeam, telescope_loc=LOC)
    with pytest.raises(ValueError, match="require polarized=True"):
        directvis.simulate_vis(fluxes=stokes(3), polarized=False, **kw)
    with pytest.raises(ValueError, match="require polarized=True"):
        directvis.simulate_vis(fluxes=stokes(3, ntimes=True), polarized=False, **kw)
    with pytest.raises(ValueError, match="fluxes has shape"):
        directvis.simulate_vis(fluxes=np.ones((3, FREQS.size + 1)), **kw)

    # (Nsrc, Nfreqs, 4) with Ntimes == Nfreqs == 4 is ambiguous; read as constant Stokes, with a warning
    kw4 = dict(kw, times=TIMES[:4])
    with pytest.warns(UserWarning, match="fits both"):
        a = directvis.simulate_vis(fluxes=stokes(3), polarized=True, **kw4)
    b = directvis.simulate_vis(fluxes=np.repeat(stokes(3)[:, None], 4, axis=1), polarized=True, **kw4)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_fftvis_only_kwargs(ants_flat, uvbeam):
    ra, dec = sky()
    kw = dict(ants=ants_flat, fluxes=stokes(3, pol=False), ra=ra, dec=dec, freqs=FREQS, times=TIMES,
              beam=uvbeam, telescope_loc=LOC)
    with pytest.warns(UserWarning, match="ignores the fftvis"):
        directvis.simulate_vis(**kw, eps=1e-6, nprocesses=4)
    with pytest.raises(TypeError, match="unexpected keyword"):
        directvis.simulate_vis(**kw, not_an_option=1)
    with pytest.raises(NotImplementedError):
        directvis.simulate_vis(**kw, beam_coefs=np.ones((len(ants_flat), 1, FREQS.size)))


def test_beam_idx_checks(ants_flat, uvbeam):
    ra, dec = sky()
    kw = dict(ants=ants_flat, fluxes=stokes(3, pol=False), ra=ra, dec=dec, freqs=FREQS, times=TIMES,
              telescope_loc=LOC)
    with pytest.raises(ValueError, match="beam_idx is required"):
        directvis.simulate_vis(beam=[uvbeam, uvbeam], **kw)
    with pytest.raises(ValueError, match="entries must be in"):
        directvis.simulate_vis(beam=[uvbeam, uvbeam], beam_idx=np.full(len(ants_flat), 2), **kw)
