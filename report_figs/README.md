# report_figs — figures of the research report (Korean labels)

`figs_appendix6_ko.py` redraws the figures of 부록 Ⅵ (dynamic QRA of PVRV releases) and the matching figures of Ⅳ장 of the KOSHA research report *바이오가스 제조공정 안전관리 방안 (음식물 폐기물 처리 공정 중심)*, from the same tables the pipeline writes.

```bash
python colab/run_dynamic_qra.py --updating          # writes dynamic_qra/out
python report_figs/figs_appendix6_ko.py             # -> report_figs/out/*.png
python report_figs/figs_appendix6_ko.py --ref       # or straight from the reference tables in results/tables
```

A Korean font is needed (Noto Sans CJK KR or NanumGothic). In Colab: `!apt-get -qq install -y fonts-nanum`; the script registers the font itself.

| File | Report figure | Built from |
|---|---|---|
| `f_frame.png` | [그림 부록Ⅵ-1] 동적 정량적 위험성평가의 구성 | — (diagram) |
| `f_met.png` | [그림 부록Ⅵ-2], [그림 Ⅳ-5] 시각별·월별 대기안정도와 풍배도 | `met_hourly.csv` |
| `f_check.png` | [그림 부록Ⅵ-3] 확산계수와 축선 농도의 비교 | `consequence.py`, `dispersion_check.py` (ISC3-PG and TNO curves appear only after `colab/get_external_models.sh`) |
| `f_near.png` | [그림 부록Ⅵ-4] 근거리 플룸 축선 농도 (141 m³/h) | `consequence.py` |
| `f_far.png` | [그림 부록Ⅵ-5] 원거리 H₂S 농도 | `gaussian_farfield.py` |
| `f_ir.png` | [그림 부록Ⅵ-6], [그림 Ⅳ-6] Case 별 연간 위험도와 초기사건별 기여 | `risk_summary.csv` |
| `f_prof.png` | [그림 부록Ⅵ-7] 위험도의 월별·시각별 분포 | `risk_profiles.csv` |
| `f_static.png` | [그림 부록Ⅵ-8] 일반 빈도·정적·동적 평가와 영향별 PLL | `static_vs_dynamic.csv`, `risk_by_effect.csv`, `risk_summary.csv` |
| `f_opt.png` | [그림 부록Ⅵ-9], [그림 Ⅳ-7] 설계 대안별 최대노출 개인위험도 | `risk_summary.csv` |
| `f_tornado.png` | [그림 부록Ⅵ-10] PLL 의 단일 변수 민감도 | `risk_sensitivity.csv` |
| `f_upd.png` | [그림 부록Ⅵ-11] 운전 기록에 따른 PLL 의 베이지안 갱신 | `updating.csv` |

Tables of 부록 Ⅵ come from the same folder: <표 부록Ⅵ-2> `met_summary.csv`, <표 부록Ⅵ-3> `source_terms.csv`, <표 부록Ⅵ-4> `source_props.json`, <표 부록Ⅵ-5> `dispersion_check_*.csv`, <표 부록Ⅵ-6> `h2s_lc_table.csv`, <표 부록Ⅵ-7> `effect_distances.csv`, <표 부록Ⅵ-8> `jet_fire_check.csv`, <표 부록Ⅵ-10> `qra_params.json`, <표 부록Ⅵ-11> `risk_summary.csv`, <표 부록Ⅵ-12> `risk_by_effect.csv`, <표 부록Ⅵ-13> `static_vs_dynamic.csv`, <표 부록Ⅵ-14> `risk_summary.csv` (rows "+ direct indication", "+ second relief", "+ both").
