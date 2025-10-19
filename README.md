# Unreliable Forecasting at Turning Points: A COVID-19 Case Study

Time-series forecasts often look strong on **global error metrics** (MAE / RMSE / sMAPE) yet **break down at trend turning points (TPs)**—the exact moments when decisions are made (e.g., wave onsets, peaks, troughs, or reversals linked to policy/variants).
This repo makes those failures **visible and measurable** with a TP-aware evaluation framework and a harmonised benchmark.

## What’s in this repo

* **Standardised TP detection** for epidemic curves (peaks, troughs, abrupt slope changes).
* **Three TP-aware metrics** that complement global error:

  * **TP-aligned window sMAPE** — sMAPE measured in fixed windows centred at detected TPs.
  * **Slope direction agreement** — agreement of the local trend sign (and angle) between forecast and truth near TPs.
  * **Lead–lag timing error** — the time difference (± days) between predicted and actual TP occurrence.
* **Multi-phase evaluation** across diverse model families to expose behaviour on and around TPs.

## Models evaluated

**Baselines:** ARIMA, SARIMA, SVR, Prophet, LSTM

**Adapted from literature:** CNN-LSTM [[Zain 2021](#references)], CurveFit-LSTM [[Gomez-Cravioto 2024](#references)], DELPHI [[Li 2023](#references)], EpiLearn [[Morel 2023](#references)].

## Data

* Johns Hopkins University (JHU) CSSE COVID-19 dataset
* Google COVID-19 Community Mobility Reports
* Oxford COVID-19 Government Response Tracker (OxCGRT)

*(See each source for licences, definitions, and update cadence.)*

## Key findings (short)

1. **Sophistication ≠ robustness at TPs.** On non-stationary, real-world data, deep or intricate models do not consistently beat strong classical baselines. Among the methods, **EpiLearn**—a distance-based ensemble—shows notable promise on **slope agreement** and **lead–lag** measures, outperforming DELPHI, CNN-LSTM, and CurveFit-LSTM in our setting. This hints that cross-country incidence curves share **reusable local patterns**, inviting transfer/case-based strategies.
2. **TP-aware metrics expose hidden weaknesses.** Slope disagreement, angular deviations, and timing errors consistently **worsen near TPs**, even when global errors look fine.
3. **Drop-in compatibility.** TP-aware metrics slot naturally into standard pipelines, giving a clearer view of reliability and supporting better design and deployment choices. We advocate **routine reporting** of TP-aware summaries whenever forecasts inform trigger-based policies.

## Reproducibility (how to use)

* Install dependencies (see `requirements.txt`).
* Prepared data under data/processed.
* Run run_all.sh

## References

* Zain, A. *et al.* (2021). **COVID-19 Pandemic Forecasting Using CNN-LSTM: A Hybrid Approach.** *Journal of Control Science and Engineering*, 2021:8785636. [https://doi.org/10.1155/2021/8785636](https://doi.org/10.1155/2021/8785636)
* Morel, J-D., Morel, J-M., & Alvarez, L. (2023). **Learning from the past: A short-term forecast method for the COVID-19 incidence curve.** *PLOS Computational Biology*, 19(6):e1010790. [https://doi.org/10.1371/journal.pcbi.1010790](https://doi.org/10.1371/journal.pcbi.1010790)
* Li, M. L. *et al.* (2023). **Forecasting COVID-19 and analyzing the effect of government interventions.** *Operations Research*, 71(1):184–201. [https://doi.org/10.1287/opre.2022.2306](https://doi.org/10.1287/opre.2022.2306)
* Gomez-Cravioto, G. *et al.* (2024). **Data Analysis and Forecasting of the COVID-19 Spread: A Comparison of Recurrent Neural Networks and Time Series Models.** *Cognitive Computation*, 16:1794–1805. [https://doi.org/10.1007/s12559-021-09885-y](https://doi.org/10.1007/s12559-021-09885-y)
