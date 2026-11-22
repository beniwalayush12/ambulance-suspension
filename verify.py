"""Hand calculations vs simulation: checks used in Section 5 of the report."""
import numpy as np
from tqdm import tqdm
import model as M

ms, mu, kt, g = M.MS, M.MU, M.KT, M.G
ks = M.k_of(1.6); cs = M.c_of(0.3, ks)
out = {}

# 1. static quantities
out["k_s (N/m)"] = (1000*(2*np.pi*1.6)**2, ks)
out["sag (mm)"] = (1000*9.81/101065*1000, ms*g/ks*1000)
out["c_s (N s/m)"] = (2*0.3*np.sqrt(101065*1000), cs)
out["F_0 (N)"] = (1100*9.81, (ms+mu)*g)

# 2. undamped normal modes: hand (characteristic equation) vs simulated free vibration
a = ms*mu; b = -(ms*(ks+kt) + mu*ks); c = ks*kt          # a*w^4 + b*w^2 + c = 0
w2 = np.sort(np.roots([a, b, c]).real)
f_hand = np.sqrt(w2)/(2*np.pi)
T = 20.0; t = np.arange(0, T, M.DT); z = np.zeros(t.size)
xs, vs, xu, vu = 0.01, 0.0, 0.0, 0.0                    # body lifted 1 cm, released, no damping (track wheel: shows both modes)
zu_hist = np.zeros(t.size)
for i in tqdm(range(t.size), desc="free vibration", leave=False, mininterval=2):
    def f(xs, vs, xu, vu):
        fs = ks*(xs-xu); return vs, -fs/ms, vu, (fs - kt*xu)/mu
    k1 = f(xs, vs, xu, vu); k2 = f(xs+M.DT/2*k1[0], vs+M.DT/2*k1[1], xu+M.DT/2*k1[2], vu+M.DT/2*k1[3])
    k3 = f(xs+M.DT/2*k2[0], vs+M.DT/2*k2[1], xu+M.DT/2*k2[2], vu+M.DT/2*k2[3]); k4 = f(xs+M.DT*k3[0], vs+M.DT*k3[1], xu+M.DT*k3[2], vu+M.DT*k3[3])
    xs += M.DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); vs += M.DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    xu += M.DT/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]); vu += M.DT/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
    zu_hist[i] = xu
spec = np.abs(np.fft.rfft(zu_hist * np.hanning(t.size), 2**21)); fr = np.fft.rfftfreq(2**21, M.DT)
pk = lambda lo, hi: fr[(fr > lo) & (fr < hi)][np.argmax(spec[(fr > lo) & (fr < hi)])]
out["f_body (Hz)"] = (f_hand[0], pk(0.5, 3))
out["f_wheel (Hz)"] = (f_hand[1], pk(6, 20))

# 3. steady state on a sinusoidal road, 1 cm amplitude: hand transfer function vs simulation
def steady(fq):
    w = 2*np.pi*fq
    A = -ms*w**2 + ks + 1j*w*cs; B = ks + 1j*w*cs; C = -mu*w**2 + ks + kt + 1j*w*cs
    Xu = kt/(C - B*B/A); Xs = B/A*Xu
    acc_hand = w**2*abs(Xs)*0.01
    tt = np.arange(0, 12.0, M.DT); road = (tt, tt, 0.01*np.sin(w*tt))
    acc, ft, _ = M.simulate(ks, cs, road)
    acc_sim = np.abs(acc[tt > 8, 0]).max()
    return acc_hand, acc_sim
for fq in [1.0, 5.0]:
    out[f"body accel. amplitude at {fq:.0f} Hz (m/s^2)"] = steady(fq)

for k, (h, s) in out.items():
    print(f"{k:40s} hand {h:12.4f}   sim {s:12.4f}   diff {abs(h-s)/abs(h)*100:.3f}%")
