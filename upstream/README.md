# upstream/

Pinned copy of what this repository needs from [biogas-pvrv-BN](https://github.com/drinkgarlic/biogas-pvrv-BN), taken at commit `a466f81` (BN v4.3: input series from the 1-s model itself, per Case; v4.2 `c68d33d`: flare demanded at the full holder; v4.1 `6ec31d6`: p_pv_fouled 0.1, NOG-070 p_pv_stuck and beta_pv; `pv_bn.py` only changed since `bc655ef`).

| Path | Origin in biogas-pvrv-BN | Used by |
|---|---|---|
| `bayesian_network/pv_bn.py`, `physics.py` | `bayesian_network/` | `risk.py` runs the network hour by hour (outputs p_k, d_k, o_k, s_k) |
| `bayesian_network/fill_hourly_onoff_c{1,3}_f{100,80}.csv`, `flare_demands_onoff_c{1,3}_f{100,80}.csv` | `bayesian_network/` | hourly fill and flare start times of the four QRA cases, from the 1-s model without ignition failures (BN `make_onoff_inputs.py`) |
| `data/*.csv` | `data/` | PyADM1ODE hourly production and composition; KMA ASOS 138 hourly pressure |
| `pressure_model_results/summary_stop.csv` | `results/pressure_model/` | R2 statistics |
| `pressure_model_results/episodes_R1_f100.csv` | 1-s on/off runs, flare start 100 % | R1 chattering-episode statistics |

To move to a newer version of the network, copy the files again and note the new commit here.
