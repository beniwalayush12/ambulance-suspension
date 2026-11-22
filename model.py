"""
Ambulance suspension: simplified quarter-car study.

Vehicle (fixed):   m_s = 1000 kg, m_u = 100 kg, k_t = 350 kN/m
Passive knobs:     spring k_s, damper c_s
Smart damper:      wants force F = -(b_s*v_s - b_u*v_u); real damper can only give
                   c in [0.8, 14] kN s/m and needs ~5 ms to change
Roads:             rough road (ISO 8608 class C, 300 m, 40 km/h)  -> comfort, grip
                   bump test (7 cm pothole + 10 cm speed breaker, 30 km/h) -> peak jolt
"""
import json
import numpy as np
from tqdm import tqdm

G, DT = 9.81, 5e-4
MS, MU, KT = 1000.0, 100.0, 350e3
FST = (MS + MU) * G
B_S, B_U = 16e3, 0.8e3          # smart-damper gains (N s/m)
C_MIN, C_MAX, TAU = 800.0, 14e3, 0.005
rng = np.random.default_rng(7)


def road_profile(x, gd):
    n = np.linspace(0.011, 2.83, 600); dn = n[1] - n[0]
    amp = np.sqrt(2 * gd * (n / 0.1) ** -2 * dn)
    ph = rng.uniform(0, 2 * np.pi, n.size)
    return (amp * np.cos(2 * np.pi * np.outer(x, n) + ph)).sum(1)


def make_road(length, v, gd, bumps=False):
    t = np.arange(0, length / v, DT); x = v * t
    z = road_profile(x, gd)
    if bumps:
        m = (x > 30) & (x < 30.9); z[m] -= 0.07 * np.sin(np.pi * (x[m] - 30) / 0.9) ** 2
        m = (x > 70) & (x < 73.7); z[m] += 0.10 * np.sin(np.pi * (x[m] - 70) / 3.7)
    return t, x, z


ROUGH = make_road(300.0, 40 / 3.6, 256e-6)
BUMP = make_road(100.0, 30 / 3.6, 16e-6, bumps=True)


def simulate(ks, cs, road, smart=False):
    _, _, z = road
    ks = np.atleast_1d(ks).astype(float); cs = np.atleast_1d(cs).astype(float)
    n = ks.size
    xs = np.zeros(n); vs = np.zeros(n); xu = np.zeros(n); vu = np.zeros(n)
    c = cs.copy() if not smart else np.full(n, C_MIN)
    acc = np.zeros((z.size, n)); ft = np.zeros((z.size, n)); cl = np.zeros((z.size, n))

    def f(xs, vs, xu, vu, zr, c):
        fs = ks * (xs - xu) + c * (vs - vu)
        tyre = np.minimum(KT * (xu - zr), FST)
        return vs, -fs / MS, vu, (fs - tyre) / MU

    for i in tqdm(range(z.size), desc="simulating", leave=False, mininterval=2):
        if smart:
            rel = vs - vu; want_f = B_S * vs - B_U * vu
            with np.errstate(divide="ignore", invalid="ignore"):
                want = np.where(want_f * rel > 0, want_f / rel, C_MIN)
            want = np.clip(np.nan_to_num(want, nan=C_MIN, posinf=C_MAX), C_MIN, C_MAX)
            c = c + (want - c) * (DT / TAU)
        z0, z1 = z[i], z[min(i + 1, z.size - 1)]; zh = 0.5 * (z0 + z1)
        k1 = f(xs, vs, xu, vu, z0, c)
        k2 = f(xs + DT/2*k1[0], vs + DT/2*k1[1], xu + DT/2*k1[2], vu + DT/2*k1[3], zh, c)
        k3 = f(xs + DT/2*k2[0], vs + DT/2*k2[1], xu + DT/2*k2[2], vu + DT/2*k2[3], zh, c)
        k4 = f(xs + DT*k3[0], vs + DT*k3[1], xu + DT*k3[2], vu + DT*k3[3], z1, c)
        xs = xs + DT/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); vs = vs + DT/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        xu = xu + DT/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]); vu = vu + DT/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        acc[i] = -(ks * (xs - xu) + c * (vs - vu)) / MS
        ft[i] = -np.minimum(KT * (xu - z1), FST)
        cl[i] = c
    return acc, ft, cl


def wk(acc):
    """ISO 2631-1 W_k frequency weighting (vertical whole-body vibration)."""
    f = np.fft.rfftfreq(acc.shape[0], DT); s = 2j * np.pi * f
    w1, w2, w3, w5, w6 = [2*np.pi*v for v in (0.4, 100, 12.5, 2.37, 3.35)]
    q = 1/np.sqrt(2)
    H = (s**2/(s**2 + w1*s/q + w1**2)) * (w2**2/(s**2 + w2*s/q + w2**2)) \
        * ((1 + s/w3)/(1 + s/(0.63*w3) + s**2/w3**2)) * ((s**2 + w5*s/0.91 + w5**2)/(s**2 + w6*s/0.91 + w6**2))
    return np.fft.irfft(np.fft.rfft(acc, axis=0) * H[:, None], acc.shape[0], axis=0)


def evaluate(ks, cs, smart=False):
    a, ftr, cl = simulate(ks, cs, ROUGH, smart)
    ab, fb, clb = simulate(ks, cs, BUMP, smart)
    m = dict(comfort=np.sqrt((wk(a) ** 2).mean(0)),
             grip=np.sqrt((ftr ** 2).mean(0)) / FST * 100,
             jolt=np.abs(ab).max(0))
    return m, dict(ab=ab, fb=fb, clb=clb)


def k_of(f): return MS * (2*np.pi*f) ** 2
def c_of(z, k): return 2 * z * np.sqrt(k * MS)


if __name__ == "__main__":
    out = {}
    # 1. three passive set-ups
    setups = {"soft": (1.0, 0.15), "stock": (1.6, 0.30), "stiff": (2.4, 0.60)}
    ks3 = np.array([k_of(f) for f, z in setups.values()])
    cs3 = np.array([c_of(z, k) for (f, z), k in zip(setups.values(), ks3)])
    m3, s3 = evaluate(ks3, cs3)
    for j, n in enumerate(setups):
        out[n] = dict(f=setups[n][0], zeta=setups[n][1], ks=ks3[j], cs=cs3[j], sag_mm=MS*G/ks3[j]*1000,
                      **{k: float(v[j]) for k, v in m3.items()})
    # 2. smart damper on the stock spring
    ms_, ss = evaluate(ks3[1:2], cs3[1:2], smart=True)
    out["smart"] = dict(ks=ks3[1], b_s=B_S, b_u=B_U, c_min=C_MIN, c_max=C_MAX, tau=TAU,
                        **{k: float(v[0]) for k, v in ms_.items()})
    # 3. sweep: three springs x nine damping ratios = 27 runs
    F, Z = np.meshgrid([1.0, 1.6, 2.4], np.round(np.arange(0.1, 0.91, 0.1), 2))
    ksw = k_of(F.ravel()); csw = c_of(Z.ravel(), ksw)
    msw, _ = evaluate(ksw, csw)
    np.savez("sweep.npz", f=F.ravel(), z=Z.ravel(), **msw)
    np.savez("series.npz", t=BUMP[0], x=BUMP[1], z=BUMP[2], acc=s3["ab"], ft=s3["fb"],
             s_acc=ss["ab"][:, 0], s_ft=ss["fb"][:, 0], s_c=ss["clb"][:, 0])
    st, sm = out["stock"], out["smart"]
    out["change_pct"] = dict(comfort=(sm["comfort"]/st["comfort"]-1)*100, jolt=(sm["jolt"]/st["jolt"]-1)*100,
                             grip=(sm["grip"]/st["grip"]-1)*100)
    json.dump(out, open("results.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {kk: round(float(vv), 3) for kk, vv in v.items()})
