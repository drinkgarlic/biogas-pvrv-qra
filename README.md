# biogas-pvrv-qra

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drinkgarlic/biogas-pvrv-qra/blob/main/colab/Biogas_PVRV_QRA_Colab.ipynb)

**Dynamic quantitative risk assessment of pressure/vacuum relief valve (PVRV) releases at a food-waste biogas plant.**

This repository asks what a venting PVRV means for the people who work on the digester roofs, hour by hour. It takes the hourly opening probabilities of the Bayesian network in [biogas-pvrv-BN](https://github.com/drinkgarlic/biogas-pvrv-BN), turns each opening into a release, a plume, an effect and an exposure, and integrates the result over one year of plant operation and weather. The answer is not a single number but a profile — by month, by hour of the day, by initiating event — whose priors can be updated as operating records accumulate.

The case plant is the municipal food-waste digestion facility in Pohang, Korea: two digesters (400 m³ headspace each, 35 °C), a 500 m³ double-membrane gas holder, DN200 weight-loaded PVRVs set at 5.0 kPag on the digester roofs (vent outlet 18.25 m above grade), a 60 m³/h consumer and a 150 m³/h flare. Gas composition comes from a PyADM1ODE simulation of the same year: CH₄ 64 %, CO₂ 30 %, H₂S 2,500 ppm (930–5,850), relative density 0.80.

---

## 1. Repository layout

```
biogas-pvrv-qra/
├── README.md, LICENSE (MIT), requirements.txt
├── data/
│   └── kma_asos138_pohang_hourly_met.csv   KMA ASOS 138 hourly wind, direction, temperature, cloud, ceiling, radiation (UTF-8, English headers)
├── upstream/                          pinned inputs from biogas-pvrv-BN (see upstream/README.md for the commit)
│   ├── bayesian_network/              pv_bn.py (v4, with o_k / s_k outputs), physics.py, on/off operating-model series
│   ├── data/                          PyADM1ODE hourly output, KMA ASOS 138 hourly pressure
│   └── pressure_model_results/        the tables of the 1-s model that the source terms need
├── dynamic_qra/
│   ├── met_stability.py               stage 0: hourly Turner stability class, vent-height wind, calms, wind rose
│   ├── source_terms.py                stage 2: release types R1-R5
│   ├── consequence.py                 stage 3-4: near-field plume, calm plume, R5 transient, H2S probit
│   ├── gaussian_farfield.py           stage 3: far-field Gaussian, H2S endpoints, model comparison
│   ├── dispersion_check.py            consistency of the dispersion models (own and GitHub)
│   ├── jet_fire_check.py              flame regime of an ignited release (jet fire or buoyant vent flame)
│   ├── risk.py                        stage 5-6: exposure model, event tree, hourly risk integration, design options, tornado
│   ├── updating.py                    stage 7: Bayesian updating of the flare PFD and the foaming rate
│   └── fig_qra.py                     figures
├── colab/
│   ├── run_dynamic_qra.py             one-command driver
│   ├── get_external_models.sh         clones the GitHub dispersion codes used as cross-checks into external/
│   └── Biogas_PVRV_QRA_Colab.ipynb
└── results/
    ├── tables/                        reference outputs of the study
    └── figures/
```

## 2. Quick start

```bash
git clone https://github.com/drinkgarlic/biogas-pvrv-qra.git
cd biogas-pvrv-qra
pip install -r requirements.txt
bash colab/get_external_models.sh                 # optional, for dispersion_check.py
python colab/run_dynamic_qra.py                   # ~3 min
python colab/run_dynamic_qra.py --updating        # + Bayesian updating demonstration (~6 min)
```

Weather comes from `data/kma_asos138_pohang_hourly_met.csv`; a new KMA download can be converted with `python dynamic_qra/met_stability.py --convert <raw.csv>`. Outputs go to `dynamic_qra/out` and `dynamic_qra/fig`. Nothing outside this repository is needed: `upstream/` carries the network, its inputs and the pressure-model tables. If you have re-run the 1-s pressure model in biogas-pvrv-BN, point `PVRV_PRESSURE_OUT` at its `pressure_model/out` folder and the source terms are rebuilt from the full one-second records instead of the frozen tables.

---

## 3. Pipeline: which model at which step

| Stage | Question | Model used | Status |
|---|---|---|---|
| 0. Inputs | production, composition, weather per hour | PyADM1ODE hourly output; KMA ASOS 138 hourly record → Turner stability class, wind at the vent height (ISC rural power law), calms | done |
| 1. Frequency | P(PVRV open), P(operator arrives while venting), state occupancy, per hour and initiating event | Bayesian network v4 (one-hour slice, exact enumeration), pinned from biogas-pvrv-BN | done |
| 2. Source term | what comes out, how fast, for how long | 1-s lumped pressure model: R1 chatter (141 m³/h pulses, duty 0.31), R2 downstream stop, R3 isolated digester; no relief (PVRV fouled by foam, or hardware fail-to-open) → seal failure at 8 / 10 / 12 kPag | done; H₂S of the hour from PyADM1ODE |
| 3a. Near field (0–10 m, the roof) | concentration where the operator stands | bent-over buoyant integral plume (Briggs rise, plume entrainment in quadrature with PG spread), Morton–Taylor–Turner plume for calm air | done; ground-reflection form under review |
| 3b. Far field (10 m – 1 km) | H₂S endpoints beyond the roof, odour | Gaussian plume with **ISC3 Pasquill–Gifford** dispersion coefficients and Briggs plume rise | Briggs-rural sigmas now; ISC3-PG to replace them |
| 3c. Transient (R5) | 39 m³ released in seconds | 30-s transient release; Gaussian puff as upper bound | done |
| 4. Effects | what hurts | flash fire inside the LFL envelope; **RIVM (2018) H₂S probit** (C in mg/m³) with the pulsed dose; H₂S and flash fire in the 30-s seal-failure cloud; seal failure (mechanical); buoyant vent flame (Heskestad + point source) as an equipment/escalation path | done; vent fire not yet in the event tree |
| 5. Exposure | who is there | detection → diagnosis → action per initiating event; 2 h/week routine roof work; three shifts | done |
| 6. Integration | risk per hour | PLL, IR of the most exposed individual, LSIR at 3 m, CH₄ emission, by effect; static QRA on the same scenarios; design options A/B | done; LSIR as a point, not a contour |
| 7. Dynamic updating | how does it move as records arrive | Beta–binomial (flare PFD), Gamma–Poisson (foaming), synthetic five-year records | done |
| 8. Uncertainty | how far can the judgement move it | one-at-a-time tornado | Monte Carlo propagation next |

Models screened out, with the reason recorded in the code: **jet fire** (flame Froude number 0.005–0.017 at the outlet, `jet_fire_check.py`), **dense-gas models** such as HEGADAS / HeavyGas.jl (the Richardson number is negative for a gas lighter than air; dense only below ~52 % CH₄ at 35 °C), **PHAST** as the primary model (kept for a few cross-check points only, so the result does not depend on a black box).

---

## 4. How the code works

**`source_terms.py`** — classifies the releases into five types and extracts their statistics. R1, the chattering release after a flare ignition failure with the holder full (per-episode duration, cycles, volume and composition from the 1-s runs: median 11.5 min, 23 cycles, 8 m³, 141 m³/h while the valve is open); R2, the continuous release after a downstream stop (production rate, 84 m³/h); R3, one digester isolated by foam, condensate, freezing or a shut-off valve (that digester's production, 41 m³/h, until the operator responds); R4, the stuck-vent-valve ratchet (a few m³); and R5, a fouled or stuck PVRV with no relief, in which the digester pressure rises at `k_iso · q/2 ≈ 11 kPa/h` until a roof seal fails (assumed at twice the design pressure, 27 min after the blockage) and 39 m³ of compressed headspace escapes followed by q/2 at roof level. Composition from PyADM1ODE: CH₄ 64 %, H₂S 2,500 ppm (930–5,850), relative density 0.80, LFL of the mixture 7.8 vol %.

**`consequence.py`** — three analytical models, written so that PHAST or ALOHA can be used as a cross-check: a bent-over buoyant plume for continuous releases in wind (Pasquill–Gifford spread combined in quadrature with the plume's own entrainment, Briggs rise), a Morton–Taylor–Turner plume for calm air, and a 30-s transient treatment of the R5 depressurisation (the TNO instantaneous puff is reported as an upper bound). Effects: a person inside the LFL contour at ignition (flash fire) and the RIVM H₂S probit. The LFL plume reaches 1–3.3 m from the vent and is 0.1–0.4 m wide; H₂S on the axis is 200–1,400 ppm at 1 m and 30–220 ppm at 3 m. The hazard is therefore confined to the roof, and above all to the person who stands at the valve.

**`risk.py`** — the event tree and the integration. `QRA` lists every parameter with its source or the word "assumption". The exposure side is built around what the operator can actually see: for every initiating event the dictionary states the detection channel (a flare ignition-failure alarm, a consumer trip, the PAH on PT-101 of the affected digester at 4.8 kPag, nothing at all for a stuck vent valve — the PVRV itself has no position switch and there is no gas detector at the vent), the action that follows (a control-room restart, a manual flare start, a field check of the digester on its roof, an inspection at the holder) and, from those two, the probability that the response puts someone on the digester roof while the valve is still venting: 0.10 / 0.05 (day / night) after an ignition failure or a downstream stop, 0.70 / 0.50 after a PAH, 0.30 / 0.20 for the vent-valve branch. Routine presence is 2 h per week of regular work on the digester roofs in the day shift, 30 % of it within a few metres of the valve, with a fatality per hour of coincidence that honours the duty cycle. A PVRV that does not relieve has its own timing sub-model (`overpressure_chain`). Two causes feed it: foam that fouls the valve (the BN parameter `p_pv_fouled` = 0.1, one value for the BN and the QRA) and a hardware fail-to-open taken from NOG-070 (2020), λ_DU = 1.1 × 10⁻⁶ /h with an annual proof test, PFD = 4.8 × 10⁻³ per valve — 1oo2 with β = 0.1 when the holder is full (both valves see the shared header), one valve when a digester is isolated. The window from the PAH to the seal failure is computed hour by hour from the rise rate (k_iso·q(h)/2 for an isolated digester, k_full·q_net(h) on the shared header) for a seal that lets go at 8, 10 or 12 kPag (0.25 / 0.5 / 0.25); the arrival time follows a continuous response curve through the BN buckets (0 at 5 min, then R(≤ 0.5 h), R(≤ 2 h), R(≤ 8 h)); on the shared header an automatic flare retry or the end of the downstream stop can also relieve the pressure. The manual vent is on the roof, is fouled by the same foam with probability 0.3 (not for the other causes), and opening it takes 3 min — which gives the probability of preventing the failure, of standing beside an open manual vent, and of being on the roof when the seal lets go (mechanical injury 0.1, the 30-s cloud with flash fire and H₂S). Three shifts share the alarms, so the script reports both the PLL over everyone who responds or works on the roof and the IR of the most exposed individual (one third of the alarms plus all the routine work). The BN is run inside with two extra outputs added for this purpose: `o_k(h)`, the probability that the operator reaches the valve while it is still venting, and `s_k(h)`, the state probability by initiating event. The script writes the hourly risk, the annual summary (with two design options: A, a position switch or fixed gas detector that tells the control room the valve is open; B, an independent second relief path and a remotely operated vent, common-cause factor 0.1), the PAH warning times, the monthly and hour-of-day profiles, a breakdown by effect, a **static QRA on the same scenarios** (`static_vs_dynamic`: the same annual event numbers with weather, day/night exposure and response, q and H₂S replaced by their annual means, and one factor at a time), a generic-frequency QRA (1 × 10⁻² spurious openings per valve-year, no time structure) and a tornado over both the QRA and the BN parameters.

**`updating.py`** — Beta and Gamma conjugate updating of the flare PFD and the foaming rate with five years of synthetic records drawn at three "true" values, re-evaluating the risk with the posterior mean and the 5–95 % quantiles.

**`dispersion_check.py`** — runs every sigma scheme, ours and the ones pulled from GitHub, through one and the same Gaussian, compares them with the tabulated Pasquill–Gifford–Turner values, checks the mass balance and the units of the external routines, and documents the near-field reflection question. **`jet_fire_check.py`** — flame regime, size, tilt and radiation of an ignited release.

---

## 5. What it found


| Case | Flare start | PVRV openings /yr | PLL, responders + routine /yr | IR, most exposed individual /yr | ignition-failure chatter (PLL) | digester isolation (foam, condensate, freezing) | no relief → manual vent / seal failure (fouled + hardware) | routine roof work | H₂S share of PLL | CH₄ vented kg/yr |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (4.0 kPag) | 100 % | 79.0 | 3.6 × 10⁻² | 1.3 × 10⁻² | 2.0 × 10⁻² | 1.1 × 10⁻² | 3.8 × 10⁻³ | 1.3 × 10⁻³ | 97 % | 323 |
| 1 (4.0 kPag) | 80 % | 1.40 | 1.6 × 10⁻² | 5.5 × 10⁻³ | ~0 | 1.1 × 10⁻² | 3.8 × 10⁻³ | 4.5 × 10⁻⁴ | 94 % | 69 |
| 3 (2.0 kPag) | 100 % | 2.83 | 1.5 × 10⁻² | 5.1 × 10⁻³ | ~0 | 1.1 × 10⁻² | 3.8 × 10⁻³ | 1.8 × 10⁻⁴ | 94 % | 70 |
| 3 (2.0 kPag) | 80 % | 1.31 | 1.5 × 10⁻² | 5.1 × 10⁻³ | ~0 | 1.1 × 10⁻² | 3.8 × 10⁻³ | 1.7 × 10⁻⁴ | 93 % | 64 |

**v1.4 (2026-10-30): hourly gas composition and CO₂ toxicity.** CH₄, CO₂ and H₂O of the raw gas now enter hour by hour (PyADM1ODE), as H₂S already did: the composition sets the buoyancy of the vented gas and its LFL (composition bins of 1 % CH₄). CO₂ toxicity is added with the HSE dangerous toxic loads (n = 8, SLOT 1.5×10⁴⁰, SLOD 1.5×10⁴¹ ppm⁸·min) written as a probit (SLOT = 1 %, SLOD = 50 %) and combined with H₂S as independent action, P = 1 − (1 − P_H₂S)(1 − P_CO₂); oxygen at 1 m is reported (lowest about 10 % on the calm-air axis) but not given a fatality of its own. The daytime radiation cross-check of the Turner classes now uses the EPA SRDT key (EPA-454/R-99-005, table 6-7: 925 / 675 / 175 W/m²): exact agreement 66 %, within one class 97 %. Results move by about 1 %: PLL 3.67 / 1.58 / 1.52 / 1.51 × 10⁻², IR_worker 1.31 × 10⁻² / 5.59 / 5.19 / 5.16 × 10⁻³; CO₂ adds 0.3–0.4 % of PLL. It matters where H₂S is reduced: with in-situ desulphurisation (H₂S × 0.2) the Case 1 / 100 % PLL is 3.5 × 10⁻³ with CO₂ and 2.3 × 10⁻³ without, and with the Bevi 2009 H₂S probit 2.8 / 1.8 × 10⁻³ (Case 1, 100 / 80 %) instead of 1.6 / 1.1 × 10⁻³. Static vs dynamic: −5 to +3 %. New sensitivity rows: CO₂ toxicity off, composition at its annual mean.

**v1.3 (2026-10-27): BN v4.3 pinned.** The BN now reads the hourly holder fill and the flare start times of the 1-s pressure model itself (a year without ignition failures, per Case), so the two packages share one holder and one operating logic. The former operating-model series demanded the flare 2,103 times a year in every case; the 1-s series give 1,997 (Case 1) and 2,037 (Case 3) at 100 %, 2,004 and 2,043 at 80 %. Only the 100 % cases move: Case 1 openings 83.1 → 79.0, CH₄ 336 → 323 kg, PLL 3.73 → 3.62 × 10⁻², IR 1.33 → 1.29 × 10⁻², ignition-failure branch 2.06 → 1.96 × 10⁻²; Case 3 openings 2.88 → 2.83. The 80 % cases are unchanged.

**v1.2 (2026-10-26): BN v4.2 pinned.** The BN now demands the flare when the holder is just full instead of at 99.9 % fill. Only the flare-start-100 % cases move: Case 1 openings 81.5 → 83.1, CH₄ 318 → 336 kg, PLL 3.68 → 3.73 × 10⁻², IR 1.31 → 1.33 × 10⁻²; the 80 % cases are unchanged.

**v1.1 (2026-10-21): the H₂S probit.** Up to v1.0 the H₂S fatality used the Bevi (2009) constants (−11.5, 1, 1.9) with the concentration in **ppm**, whereas they are defined for mg/m³; the same constants with the right unit give about twice the toxic load (1.417^1.9), and the RIVM (2018, interim) probit Pr = −7.87 + 0.31 ln(C^6.52 t) (C mg/m³, t min; 30-min LC50 ≈ 244 ppm, LC01 ≈ 77 ppm) is much steeper still. The v1.0 results (PLL 1.1–1.3 × 10⁻³, IR 3.8–5.1 × 10⁻⁴, blocked-PVRV chain 78–93 %) were therefore driven by an H₂S term that was close to zero. With the RIVM probit the plume at 1 m from an open valve (several hundred to 1,400 ppm H₂S) is lethal within one to two minutes of the pulsed exposure, and H₂S becomes 93–97 % of the PLL. The Bevi (2009) value with the correct unit gives 1.1–1.5 × 10⁻³ — the choice of probit is now the largest single uncertainty, and RIVM itself calls its 2018 function interim. The same revision unified the PVRV fouling probability (BN 0.3 → 0.1, now read from the BN), added the NOG-070 hardware fail-to-open branch, made the seal-failure window depend on the hour, used the hourly H₂S of the raw gas, added H₂S in the seal-failure cloud, and added the static QRA, the near-field and entrainment sensitivities.

**Reliability data behind the no-relief and flare branches.** The PVRV fail-to-open rate is the NOG-070 (2020) Table A.2.1 value for PSVs, λ_DU = 1.1 × 10⁻⁶ /h for "fail to open before test pressure" (2.2 × 10⁻⁶ /h for all DU modes as a sensitivity), with a factor 5 for dirty service; no manufacturer datasheet of the plant's DN200 weight-loaded PVRV was available, so the valve characteristics (set 5.0 kPag, 1,410 m³/h at 110 %, minimum stable opening 10 %) follow a representative public catalogue of the same type and API 2000 / ISO 28300 — a limitation to be closed with the actual datasheet. The flare PFD stays at 0.05 (0.01–0.10 in the tornado): the per-component flare values of Taleb-Berrouane & Lounis (2016) are basic events of a fault tree, not a per-demand probability, and a 2026 field study of flares found a failure share of 7.6 % against the 0.2 % usually assumed in the literature, so a value in the SIL 1 band is the defensible middle.

**Static vs dynamic on the same scenarios** (`static_vs_dynamic.csv`, `qra_static_dynamic.png`). With the same annual event numbers and every time-dependent factor at its annual mean, the static QRA gives 3.6 × 10⁻² / 1.7 × 10⁻² / 1.6 × 10⁻² / 1.6 × 10⁻² — within −7 % to +1 % of the dynamic result. Weather, day/night exposure, the hourly q and the hourly H₂S each move it by less than 10 %, partly in opposite directions (releases come more often at night, when fewer people climb, and in months with lower H₂S). The three orders of magnitude between the dynamic QRA and the generic-frequency QRA (2.4 × 10⁻⁵) therefore come from the scenario frequencies the BN derives from the operating mode, not from averaging over time. The time profile is still what tells the plant *when* the risk is carried (the hourly and monthly figures).

Weather is the KMA hourly record (`met_stability.py`). The Turner method puts 36 % of the year in class D, 28 % in F, 10 % in E and 25 % in A–C; calm air (10-m wind below 0.5 m/s) is only 1.2 % of the hours, against the 20 % the earlier day/night mix assumed. Against that day/night mix the KMA record lowers the PLL by about 15 % (tornado); calm air, the case that gives 1,400 ppm at 1 m, is rare, and the stable F hours carry most of the H₂S fatality. The record has a gap of 24 h at the end of the year (the download stops at 2021-10-31 00:00), filled with the previous day and flagged.

Tying the response to what the instruments show changed the composition of the result more than its size. A chattering valve after an ignition failure no longer sends anyone to the digester roof — the alarm is at the flare, the fix is a restart from the control room — so only one alarm in ten (one in twenty at night) sends someone up while the valve chatters. With 79 openings a year that is still enough to make the chatter branch the largest term of the 100 % case (2.0 × 10⁻² PLL), because every such visit carries a 1–2 % chance of a lethal H₂S dose. In every case the digester-isolation branches (foam, condensate, freezing, shut-off valve) are next: the PAH sends the operator to that digester's roof with probability 0.7 / 0.5, while the valve is venting (1.1 × 10⁻² PLL, flat across the cases). A PVRV that does not relieve adds 3.8 × 10⁻³ — 3.6 × 10⁻³ from foam fouling (0.047 seal failures a year) and 2.3 × 10⁻⁴ from the hardware fail-to-open (0.005 a year; the shared header is protected by two valves and by the automatic retry, an isolated digester by one valve). The generic-frequency QRA stays three orders of magnitude below the dynamic one, which is the HAZID finding ("PVRV release frequency is underestimated by generic data") in numbers.

The tornado ranks the assumptions. For both cases the result moves most with the probability of a prolonged stay in the H₂S plume (0.005 with a personal detector and a procedure, 0.3 without: 1.2 × 10⁻² → 0.49 PLL for the 100 % case), the probit (Bevi 2009: 1.5 × 10⁻³), the H₂S content of the gas (in-situ desulphurisation, ×0.2: 2.5 × 10⁻³ / 1.6 × 10⁻³), the share of alarms that put someone on the roof, the PAH (without it 7.9 × 10⁻² / 3.8 × 10⁻²) and the near-field concentration (free plume ×0.5 → recirculation ×2: 1.6–4.8 × 10⁻²); the flare PFD and the hard-fault share matter for the 100 % case only. The fouling probability (0.03–0.3), the NOG-070 rate (2.2 × 10⁻⁶ /h), dirty service (×5) and β (0.05–0.2) move the result by less than 40 %. The PAH warning times are in `alarm_warning_times.csv`: with the holder full the alarm comes 2–4 min before the valve opens whatever the case, and when a digester is isolated the margin is under a minute — the alarm tells the operator that the valve is about to open, not that it has.

Two design options are costed in risk terms (`qra_design_options.png`, IR of the most exposed individual). A, a position switch or a fixed CH₄/H₂S detector at the PVRV, removes the unplanned climbs to a venting valve (roof presence while venting × 0.2, plume stay × 0.25) and acts on the dominant H₂S term: 1.3 × 10⁻² → 2.1 × 10⁻³ (100 %), 5.1–5.5 × 10⁻³ → 1.5–1.6 × 10⁻³ (others). B, an independent second relief path on its own nozzle with a remotely operated vent, removes most of the no-relief chain but leaves the H₂S exposure: 1.2 × 10⁻² and 3.9–4.3 × 10⁻³. Only A + B brings every case below the HSE worker line: 9.0 × 10⁻⁴ (100 %) and 2.7–3.5 × 10⁻⁴. In-situ desulphurisation of the digester headspace (H₂S × 0.2) is a third lever of the same order.

The far-field Gaussian (`gaussian_farfield.py`) places the H₂S endpoints beyond the roof: at the vent height AEGL-2 is reached within 2–3 m and AEGL-1 within 15–40 m in neutral weather, while at ground level nothing exceeds AEGL-1 and the odour threshold (0.01 ppm) extends to 150–1,300 m depending on stability — a nuisance, not a hazard.

**Consistency of the dispersion models** (`dispersion_check.py`, `dispersion_check.png`). Before any of the pulled codes was used as a reference it was read and run under controlled conditions, and the first version of the comparison turned out to be wrong in two places: the external schemes had been evaluated with the ground-level form Q/(π u σy σz), twice the elevated-source form used for our own Gaussian, and the TNO package had been called with the vent height as its `z0`, which is in fact the roughness length (its docstring says "source height"; its own examples pass `z0 = 0.2` next to a separate `hs`). Both are corrected. What the check now says: the Briggs (1973) open-country sigmas used here and the EPA ISC3 fits of the Pasquill–Gifford–Turner curves in `koaandmud/GPM` reproduce the tabulated PG values at 100 m and 1 km within 20–30 % (σz in class F is the widest gap, 1.6 vs 2.2 m at 100 m) and, run through one and the same Gaussian, agree within a factor of 1.5 beyond 10 m; below 100 m both are extrapolations, since the PG curves are not defined there. The TNO 'farm' scheme in `gerritjandebruin/gaussian-plume` is not a 10-min-average scheme at all: its σy grows with the travel time to the power 0.35 — a relative-diffusion (instantaneous plume) parameterisation from mobile transect measurements of farm emissions — and with the correct `z0` it gives plumes four times narrower and axis concentrations 3–7 times higher than the PG schemes. It is therefore reported but not used; its concentration routine also returns ppb for a source in g/s, which its documentation does not say. Two further GitHub implementations of the same ISC3 fits were checked number by number: `Hammerling-Research-Group/FastGaussianPuff` (MIT, C++ puff model for methane monitoring) carries the identical coefficient set except one exponent beyond 10 km (0.4617 vs 0.46713 for class E), and `yunhal/CH4_dispersion` (MIT) reproduces the near field exactly but has transcription errors in the distance breakpoints and in two σy constants (class D σz +4 % and class F σz +9 % at 1 km). EPA's AERMOD itself is distributed from the SCRAM site, not from GitHub, and the GitHub `aermod` package is only an R wrapper around the Windows executable. The mass balance ∫∫ u C dy dz = Q closes to 1.000 for every scheme. Two things the check turned up about our own near-field plume are recorded for decision: it uses the fully reflected form Q/(π u σy σz), which is twice the free-plume value whenever σz is smaller than the outlet height above the roof — at 1–10 m from the vent that is always, so the near-field concentrations (and with them the H₂S dose, ∝ C^1.9) carry a factor-2 conservatism and the LFL distance 2.1 m would be 1.3 m in the reflected form; and in class F the Pasquill–Gifford sigmas are millimetres at 1–3 m, which is why the near field needs the integral plume in the first place. HEGADAS-type dense-gas models (`HeavyGas.jl`, an HGSYSTEM port) cannot run for this gas: their Richardson number is g H (1 − ρa/ρm)/u*², negative for a mixture lighter than air, and the vented gas is lighter than air (relative density 0.80–0.91 at 35 °C) unless the methane content drops below about 52 %.

**Is an ignited release a jet fire?** No (`jet_fire_check.py`). The valve relieves at 5 kPag; the gas does reach about 60 m/s in the 1–2 mm annular gap under the pallet, but that jet is turned and slowed inside the valve body and the weather hood and would make a flame of 0.1–0.4 m even if it were free. What leaves the DN200 outlet moves at 0.4–1.2 m/s, the flame Froude number is 0.005–0.017 (Delichatsios: buoyancy-controlled below 1, jet fire above 5) and stays at 0.17 even at the rated 1,410 m³/h the plant never reaches. An ignited release is therefore a buoyant vent flame — 0.2–0.8 MW, 1.9–3.2 m long (Heskestad), bent 40–60° by a 3–5 m/s wind — and jet-fire correlations (Chamberlain, API 521 jet) are outside their range here. Its radiation does not reach a person: 12.5 kW/m² within 0.9 m of the flame centre, about 3.7 kW/m² at 1 m beside the vent, no fatality by the Eisenberg probit even for 30 s. What a vent fire does threaten is the equipment: a flame that keeps burning on a chattering valve (outlet velocity 1.5–5 times the 0.25 m/s burning velocity of the mixture, zero when the pallet seats) heats the seat, the pallet seals and the hood, and a tilted 3-m flame can reach whatever stands downwind on the roof. That is an escalation path — a damaged PVRV that leaks or sticks feeds the blocked-PVRV chain — not an individual-risk term, and it is the reason PVRVs on biogas and tank vents are fitted with endurance-burning-proof end-of-line flame arresters.

### 5b.4 What to do with it

The event tree turns the BN's recommendations into risk terms, and it adds two of its own. Start the flare at 80 % or lower and keep the automatic retry, as before. Give the control room a direct indication that a PVRV is open (a position switch, or a fixed detector at the vent), because today the only sign is a PAH that precedes the opening by minutes, and write the procedure so that nobody climbs to a valve that is venting without a detector and a reason. Treat the vented gas as toxic first and flammable second: with raw-gas H₂S of 1,000–6,000 ppm the plume within a few metres of an open valve is lethal in minutes, so personal H₂S detectors, a no-approach rule for a venting valve, and in-situ desulphurisation are worth more than any ignition control. Take the manual vent off the roof with a second relief device on an independent nozzle or a remotely operated vent valve. Together these lines are worth more than any operating-pressure change.

---

## 6. Consistency of the dispersion models

| Code | What it is | Verdict |
|---|---|---|
| this work, Briggs (1973) open-country | PG-type sigmas, smooth in x | within 20–30 % of the PG-Turner table at 100 m and 1 km |
| [koaandmud/GPM](https://github.com/koaandmud/GPM) | EPA ISC3 fits of the PG curves | matches the table; reference implementation of ISC3-PG |
| [Hammerling-Research-Group/FastGaussianPuff](https://github.com/Hammerling-Research-Group/FastGaussianPuff) (MIT) | C++ Gaussian puff for methane monitoring | identical ISC3 coefficient set; one exponent differs beyond 10 km (class E) |
| [yunhal/CH4_dispersion](https://github.com/yunhal/CH4_dispersion) (MIT) | Gaussian puff/plume for synthetic CH₄ plumes | ISC3 with transcription errors in distance breakpoints and two σy constants (+4 / +9 % σz at 1 km) |
| [gerritjandebruin/gaussian-plume](https://github.com/gerritjandebruin/gaussian-plume) (TNO) | relative-diffusion scheme, σy ∝ travel time^0.35 | an instantaneous-plume parameterisation; not comparable with 10-min averages, reported only |
| [barche/HeavyGas.jl](https://github.com/barche/HeavyGas.jl) | HEGADAS-S port | not applicable: gas lighter than air |

Run through one formula, Briggs and ISC3 agree within a factor of 1.5 beyond 10 m; the mass balance closes to 1.000 for every scheme. Below 100 m every Pasquill–Gifford scheme is an extrapolation, which is why the near field is handled by the integral plume and only matched to the Gaussian further out.

---

## 7. Roadmap

1. ~~**Hourly weather.**~~ Done: KMA ASOS 138 → Turner class, vent-height wind and calms per hour. Next on this item: the wind direction is not used yet (it enters with the LSIR contours, item 7).
2. **Toxic pathway.** ~~RIVM probit for H₂S with the right unit, breakdown by effect, results without a personal detector~~ (v1.1). Next: H₂S + CO₂ toxic load (HSE SLOT/SLOD, RIVM probit for H₂S), O₂-deficiency zone, AEGL-2 and occupational-exposure-limit endpoints, results with and without a personal detector, and a breakdown by effect (toxic / fire / mechanical).
3. **ISC3-PG in the far field.** An own implementation of the EPA ISC3 coefficients (public tables), verified against GPM and FastGaussianPuff, replaces the Briggs-rural sigmas; Briggs stays as a sensitivity case.
4. **Near-field / far-field matching.** The integral plume up to ~10 m, then ISC3-PG from a virtual source whose initial σy0, σz0 equal the plume width at the junction; decide the ground-reflection form of the near-field plume (currently 2× the free plume).
5. **Building downwash screening** for a vent 1 m above an 18 m digester roof (Huber–Snyder / Schulman–Scire criteria as in ISC3).
6. ~~**Hourly composition.**~~ H₂S of the hour from the PyADM1ODE series (v1.1); CH₄ still the annual mean in the plume (LFL).
7. **LSIR contours** on a plant grid using the hourly wind direction, instead of a single point at 3 m.
8. **Vent fire branch.** Ignited, sustained release on the valve → seat and seal damage → leaking or stuck PVRV, with and without an endurance-burning flame arrester.
9. **Monte Carlo** propagation of the judgement parameters (Latin hypercube) for an IR distribution instead of a tornado.
10. **Updating with real records**: flare start log, alarm and response times, foaming events.
11. **PHAST cross-check** at a handful of points (R2 and R3, D5 and F2), then the appendix and the paper.

---

## 8. Limitations

- **The exposure model is judgement.** Detection channels and actions are read from the P&ID and the alarm list, but the probabilities that a response puts someone on the roof, the time spent at the valve, the presence in the plume and the whole blocked-PVRV timing sub-model have no site data behind them yet.
- **One weather station, one year.** Pohang ASOS 138 stands for the plant site; the Turner class and the 10-m wind are site-independent approximations of the roof-level flow, and a 24-h gap at the end of the record is filled from the previous day. The Turner and radiation/wind schemes agree within one class in 84 % of the daytime hours.
- **The near field is analytical.** No Pasquill–Gifford scheme is defined below 100 m; the integral plume carries a factor-2 conservatism from its reflection form; building wakes are not modelled.
- **The frequencies are inherited.** Everything upstream of the release — the 1-s model, the BN priors, the operator response distribution — carries the limitations listed in biogas-pvrv-BN.

---

## 9. References

1. Briggs GA. Diffusion estimation for small emissions. ATDL Contribution File No. 79. Oak Ridge (TN): Atmospheric Turbulence and Diffusion Laboratory; 1973.
2. Briggs GA. Plume rise predictions. In: Haugen DA, editor. Lectures on air pollution and environmental impact analyses. Boston: American Meteorological Society; 1975. p. 59–111.
3. Turner DB. Workbook of atmospheric dispersion estimates. Publication AP-26. Research Triangle Park (NC): US EPA; 1970.
4. Turner DB. A diffusion model for an urban area. Journal of Applied Meteorology. 1964;3(1):83–91.
5. Korea Meteorological Administration. Open MET Data Portal, ASOS hourly data, station 138 (Pohang) [Internet]. Available from: https://data.kma.go.kr
6. US Environmental Protection Agency. User's guide for the Industrial Source Complex (ISC3) dispersion models. Vol. II: Description of model algorithms. EPA-454/B-95-003b. Research Triangle Park (NC): US EPA; 1995.
7. Morton BR, Taylor GI, Turner JS. Turbulent gravitational convection from maintained and instantaneous sources. Proceedings of the Royal Society A. 1956;234:1–23.
8. Committee for the Prevention of Disasters. Guidelines for quantitative risk assessment (Purple Book). CPR 18E. The Hague: Sdu; 1999/2005.
9. Committee for the Prevention of Disasters. Methods for the determination of possible damage (Green Book). CPR 16E. The Hague: Sdu; 1992.
10. RIVM. Reference Manual Bevi Risk Assessments, version 3.2. Bilthoven: RIVM; 2009.
11. Heskestad G. Luminous heights of turbulent diffusion flames. Fire Safety Journal. 1983;5(2):103–108.
12. Delichatsios MA. Transition from momentum to buoyancy-controlled turbulent jet diffusion flames and flame height relationships. Combustion and Flame. 1993;92(4):349–364.
13. Eisenberg NA, Lynch CJ, Breeding RJ. Vulnerability model: a simulation system for assessing damage resulting from marine spills. Report CG-D-136-75. Washington (DC): US Coast Guard; 1975.
14. American Petroleum Institute. Pressure-relieving and depressuring systems. API Standard 521. 7th ed. Washington (DC): API; 2020.
15. Health and Safety Executive. Reducing risks, protecting people. Norwich: HSE Books; 2001.
16. Khakzad N, Khan F, Amyotte P. Dynamic safety analysis of process systems by mapping bow-tie into Bayesian network. Process Safety and Environmental Protection. 2013;91(1–2):46–53.
17. Villa V, Paltrinieri N, Khan F, Cozzani V. Towards dynamic risk analysis: a review of the risk assessment approach and its limitations in the chemical process industry. Safety Science. 2016;89:77–93.
18. Reinelt T, Liebetrau J. Monitoring and mitigation of methane emissions from pressure relief valves of a biogas plant. Chemical Engineering & Technology. 2020;43(1):7–18.
19. Gaida D. PyADM1ODE [Internet]. GitHub repository. Available from: https://github.com/dgaida/PyADM1ODE

---

## 10. Licence

MIT (see `LICENSE`). The pinned files in `upstream/` come from biogas-pvrv-BN under the same licence; the external dispersion codes are cloned at run time and keep their own terms.
