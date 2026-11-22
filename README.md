# Comfort vs Grip in Ambulance Suspensions

Course project for **PH 217: Classical Mechanics**, IIT Bombay (Autumn 2026).
Team: Sakshi (25B1850), Ayush (23B1810), Subrat (25B1855).

One rear corner of an ambulance (Force Traveller) is modelled as two coupled oscillators:
the body (1000 kg) and the wheel (100 kg) on a tyre spring (350 kN/m). We study how
suspension stiffness and damping trade patient comfort against tyre grip, and test a
semi-active "skyhook" damper.

**Interactive simulator:** https://capable-kashata-8d240d.netlify.app/

## Results

| Setup | Comfort a_w (m/s²) | Grip loss (%) | Peak jolt (m/s²) | Sag (mm) |
|---|---|---|---|---|
| Soft (1.0 Hz, ζ 0.15) | 0.49 | 12.2 | 7.7 | 248 |
| Stock (1.6 Hz, ζ 0.30) | 0.80 | 11.2 | 11.5 | 97 |
| Stiff (2.4 Hz, ζ 0.60) | 1.24 | 16.3 | 14.8 | 43 |
| Semi-active (stock spring) | 0.62 (−22%) | 11.2 (same) | 8.5 (−26%) | 97 |

## Files

| File | What it does |
|---|---|
| `model.py` | Road generation (ISO 8608 class C + pothole/speed breaker), RK4 quarter-car simulation, semi-active controller, ISO 2631-1 weighting, 27-run sweep. Writes `results.json`, `series.npz`, `sweep.npz`. |
| `verify.py` | Compares hand solutions (static sag, damping, normal modes, forced response) with simulator readings. |
| `figures.py` | Makes the result figures in `figures/` from the outputs of `model.py`. |
| `fig_trans.py` | Single-oscillator transmissibility plot (passive vs skyhook). |
| `simulator/index.html` | Stand-alone web simulator (same model, runs in the browser). |

## Running

```bash
pip install -r requirements.txt
python model.py      # ~1 min, writes results.json, series.npz, sweep.npz
python figures.py    # needs the outputs of model.py
python fig_trans.py
python verify.py
```

To use the simulator offline, open `simulator/index.html` in a browser.

## Use of AI

The simulation code (Python and the web simulator) was written with the help of an AI
assistant (Claude, Anthropic). We verified it by putting the parameters into our derived
formulas and comparing with the simulator's readings; all checks in `verify.py` agree to
within 0.01%.
