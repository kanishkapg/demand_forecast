# Task 2A: Forecast Depot Demand (Plan)

## Can Task 2A be done without Task 1?

**Yes.** Task 2A doesn't use any Task 1 model, label or prediction. The one link is a raw input file:

- **`Test Data/task1_test_inputs.csv` is needed as extra history, but nobody has to finish Task 1 first.** Training orders stop on 2026‑02‑14 (ISO week 7). `task1_test_inputs.csv` covers weeks 8–13 of 2026 (5,014 orders). The forecast runs from week 14 to week 23. Skipping that file loses the six weeks just before the forecast.
- **The 2026 weeks 8–13 may be slightly undercounted.** `task1_test_inputs.csv` holds only orders that were dispatched (4,924 attempted and 90 deferred, with no `not_run` rows). In training, `not_run` orders are only about 0.45% of all orders, so the missing demand is small. Mention it in the write‑up.
- **Route legs, outlet travel data, and anything about service time or lateness are not needed.**
- **Some deliverables are shared with the rest of the team:** one final notebook, one preprocessing document, one architecture diagram and one video. Agree early on how the Task 2A section, saved model files and inference cell will fit into the team notebook.

---

## What the data already shows

| Finding | Why it matters |
|---|---|
| 6 series (2 depots × 3 brands) over 10 weeks, so the submission has 60 rows. | This is a very small panel, so simple models with a clear structure will beat large ML models. |
| History has 117 weeks: 111 from training plus 6 from the Task 1 test file. | That gives about 2.25 years, including 2 previous April–May festival seasons. |
| No orders fall on Sundays. `is_operating` matters: 2026 week 16 has only 4 operating days and week 18 has 5. | Weekly volume depends on how many operating days the week has. |
| Festivals in the forecast window: New Year (13 Apr, week 16), Vesak (1 May, week 18) and Poson (30 May, week 22). | These weeks are where most forecast error will come from. |
| **In 2025, Peliyagoda Fresh reached 1,351 m³ in the week before New Year (normal is about 950), then fell to 623 m³ in the New Year week, which had 4 operating days.** Style jumped by 60–90% before the festival. | The demand surge comes in the week *before* the festival. The festival week itself drops. |
| New Year falls on a Monday in both 2025 and 2026, in week 16 both times. | The 2025 weeks 14–23 are an almost exact match for the forecast period. |
| The chilled share of Fresh volume is very stable: 0.365 ± 0.009. Only Fresh has chilled orders. | Chilled volume can be forecast as total × share instead of with a separate model. |
| Fresh outlets order every operating day. **Style outlets order once a week, always on the same weekday.** Tech outlets order irregularly, about once a week. | Style has a predictable order count. Tech is small (about 20–30 m³ a week) and noisy (CV around 0.3–0.4). |
| Average daily volume grows about 2–6% a year (Peliyagoda Fresh: 156 → 165 → 166 m³/day). | Include a slight trend or level term. |
| In 2026 week 10, Style spiked (Peliyagoda 195 m³, Kandy 120 m³) with no calendar flag. | Look into it: decide whether it's an outlier or a promotion. |

> **Environment note:** `pandas` is only installed in the conda env `hack` (`C:\Users\Asus\miniconda3\envs\hack`). The base Python and the other envs don't have it.

---

## Phase 0: Setup and framing (about 30 min)

1. Create `notebooks/task2a.ipynb`, or a Task 2A section in the team notebook, plus `models/task2a/` and `data/processed/`.
2. Define the problem exactly:
   - **Unit:** depot × brand × ISO week.
   - **Target 1:** `sum(order_volume_m3)` over all orders whose **`order_date`** falls in that ISO week.
   - **Target 2:** the same sum for orders with `temp_requirement == 'chilled'`.
   - **Forecast origin:** end of 2026 week 13. **Horizon:** 1–10 weeks ahead (weeks 14–23).
3. The booklet doesn't give the scoring metric. Use an absolute‑error metric, MAE or WAPE, as the main one, and also report RMSE. Because the scale differs so much between series (Peliyagoda Fresh is about 950, Kandy Tech about 20), use **WAPE** for comparing models and per‑series MAE for diagnosis.

---

## Phase 1: Load and clean the data

### Files to use

| File | Use |
|---|---|
| `Training Data/deliveries_train.csv` | Main demand history (2024‑01‑01 to 2026‑02‑14) |
| `Test Data/task1_test_inputs.csv` | Demand for 2026 weeks 8–13. Use only the order columns. |
| `General Data/calendar.csv` | ISO week mapping and the festival, payday, holiday and operating‑day features |
| `Test Data/task2a_test_inputs.csv` | The 60 rows to predict |
| `Submission Templates/submission_task2a.csv` | Output template |
| `General Data/outlets.csv` | Optional: outlet counts per depot and brand, and Style ordering weekdays |

**Not needed:** route legs, vehicles, district travel, service allowance, traffic and road conditions. Those are supply‑side data. Demand is what stores *ordered*, not what got delivered.

### Steps

1. **Parse dates carefully.** All raw files, the calendar and both order files, use `M/D/YYYY` (for example `1/1/2024`). Always pass `pd.to_datetime(..., format='%m/%d/%Y')` explicitly so a value can never be read as day-first.
2. Concatenate train and test orders, keeping only the shared order columns. Add a `source` column so rows can be traced.
3. **Integrity checks:**
   - `delivery_id` is unique across both files.
   - Every `order_date` matches a calendar date (already checked: 0 unmatched).
   - No negative or zero volumes, and no missing `order_volume_m3`.
   - Only Fresh has chilled orders (confirmed).
4. **Keep every order, whatever its `dispatch_status`** (attempted, deferred or not_run). The rules say all of them count as demand.
5. **Assign each order to a week by `order_date`, not `dispatch_date`.** Deferred orders have `dispatch_date ≠ order_date` in 100% of cases, so using the wrong column moves volume into the wrong week.
6. Merge `iso_year` and `iso_week` from the calendar. Don't compute them with `dt.isocalendar()`. They should agree, but the rules say to use the calendar's values.
7. Add the column `chilled_vol = order_volume_m3 if temp_requirement == 'chilled' else 0`.

---

## Phase 2: Build the modelling tables

### A. Weekly panel (the submission level)

- Group by `depot, brand, iso_year, iso_week` and compute `total_vol`, `chilled_vol`, `n_orders`, `n_outlets_ordering` and `n_order_days`.
- Reindex onto the **complete grid** of 6 series × 117 weeks so that a missing week becomes 0, not a dropped row. This matters for Tech.
- Add `chilled_share = chilled_vol / total_vol` for Fresh.

### B. Daily panel (recommended modelling level)

- Group by `depot, brand, date` and join the full daily calendar, including non‑operating days, which get volume 0.
- At daily level the calendar effects are much cleaner: `festival_ramp` builds up over 9 days, paydays fall on specific dates, and holidays remove specific days. Summing daily forecasts to weeks handles the "4 operating days" weeks without extra work.

### C. Weekly calendar features

Aggregate from `calendar.csv`, including the 10 forecast weeks:

- `n_operating_days`, `n_holidays`, `n_paydays`, `monsoon` (1 for every forecast week)
- `ramp_max`, `ramp_sum` (the sum works as a pre‑festival intensity measure), `has_festival`, `festival_name`
- `weeks_to_next_festival` and `weeks_since_last_festival`. These capture the pattern of a surge before the festival and a dip during it.

Save the outputs to `data/processed/weekly_panel.parquet` and `data/processed/daily_panel.parquet`.

---

## Phase 3: Exploratory analysis and visualization

Each plot should answer a specific question. Keep the plots that lead to a modelling decision, because they also go into the preprocessing document and the video.

| # | Plot | Question it answers |
|---|---|---|
| 1 | Weekly total volume per series, 6 small multiples, with festival weeks shaded and the train/test‑input boundary marked | What do the level, trend and spikes look like? Does the Task 1 test period continue the training series smoothly? |
| 2 | Year‑over‑year overlay: x = ISO week, one line per year, per series | Do the April–May patterns repeat? Line up 2024, 2025 and the 2026 history. |
| 3 | Event‑study plot: average volume relative to normal for weeks −3…+2 around each festival, split by brand | How big is the pre‑festival surge and the festival‑week dip, and does it differ by festival (New Year vs Vesak vs others) and by brand? |
| 4 | Daily volume vs `festival_ramp` (scatter or binned means) per brand | Is the ramp effect roughly linear, so it can be used directly as a feature? |
| 5 | Box plot of daily volume by day of week, and payday vs non‑payday | Are there day‑of‑week and payday effects? Thursday has the most orders. |
| 6 | Weekly volume vs `n_operating_days` | Do short weeks lose the demand, or is it pushed into the next week? |
| 7 | Chilled share over time, with festival weeks highlighted | Is a constant share enough, or does it rise before festivals? The Task 2B scenario says Fresh dairy and meat demand rises before festivals. |
| 8 | Weekly volume broken into `n_orders × avg_volume_per_order` | Does growth come from more orders or bigger orders? Especially useful for Style and Tech. |
| 9 | Rolling 8‑week mean and std, plus STL decomposition, per series | How much is trend, how much is festival effect, and how much is noise? |
| 10 | Investigation of 2026 week 10 Style: which outlets and which days | Is it a true outlier, one large order, or a repeatable pattern? |

---

## Phase 4: Feature engineering

The main constraint is the forecast horizon. At the origin (end of week 13) nothing after week 13 is known. **Any lag feature must be available at that point.** For a forecast h weeks ahead, lag‑1 isn't known. Choose one of these approaches:

- **Origin‑level features (recommended):** for every target week, compute features from data up to the *forecast origin*. Examples: mean of the last 4, 8 and 13 weeks, plus `horizon = h` as a feature. Train on many simulated origins (see Phase 6).
- **Lags of at least 10 weeks only**, such as the same week last year (lag 52). Simpler but weaker.

### Feature list

- **Level:** rolling mean of non‑festival weeks before the origin (4, 8 and 13 weeks), the robust median, and a trend slope over the last 26 weeks.
- **Seasonal:** same ISO week last year, and the same *festival‑relative* week last year. The festival‑relative one matters more, because festivals move between ISO weeks from year to year.
- **Calendar:** the features from Phase 2C.
- **Series identity:** `depot`, `brand`, or a one‑hot of the 6 series (for a global model).
- **Structural, optional:** the number of active outlets per series, and for Style, how many outlets' fixed order weekday is an operating day in the target week.

**Do not use:** dispatch status, vehicle or route fields, or anything that exists only after the orders were placed.

### As implemented (updated after Phase 4)

The origin-level approach was used. Output: `data/processed/features_origin.parquet`, **6,030 rows × 54 columns**.
- One row per origin × series × target week, with `h` = 1–10.
- 105 origins, from 2024 wk 13 to 2026 wk 13. The origin 2026 wk 13 gives exactly the 60 submission rows.
- 38 features. The builder cuts all outcome data at the origin before computing anything.

Where it differs from the list above:
- **Level:** computed on `vol_per_op_day` of *clean* weeks, not on weekly totals of non-festival weeks.
  - A clean week has 6 operating days, no ramp, and no festival.
  - For Style, ISO weeks 10 and 32 are also excluded, because they are recurring spikes.
  - Levels are the mean of the last 4, 8 and 13 clean weeks plus the median of the last 8.
  - Added `cv13`, `yoy8`, `opd_orders8`, `vpo8` and `outlets4`.
- **Seasonal:** the ISO-week analog and the festival-relative analog are stored as ratios to their own baseline (`ly_ratio_opd`, `fr_ratio`), not as raw volumes.
  - **New:** `fest_strength`, the peak ratio at the last full occurrence of the target's festival, averaged over the brand.
  - **New:** `ramp_x_strength = ramp_sum_operating × (fest_strength − 1)`.
- **Historical ratios** use a more robust baseline than Phase 3: the median of the last 6 clean weeks at least 3 weeks back, however far back they are. The fixed 3–8-week window was empty after the April–May festival run.
- **New features:**
  - `dow_weight`: the sum of day-of-week factors over the target's operating days.
  - `fest_day_closed` and `near_festival`.
  - Chilled-share features: `share_recent6`, `share_ly_target`, `share_seasonal`.
- **Targets stored:** `y_total`, `y_chilled`, `y_share`, plus the ratio targets `y_ratio_opd` and `y_ratio_week`, both relative to `lvl_med8` at the origin.
- **Leakage test:** every outcome after the origin is replaced with random numbers and the features must not change. It passes for 4 origins, including the real one.

What the features showed. These findings change Phase 5:
- **Level:** the mean of 8–13 clean weeks is best. Extrapolating the trend makes every brand worse, and the error does not grow from h = 1 to h = 10.
  - Level-only WAPE on clean weeks: Fresh 4.0%, Style 5.3%, Tech 30%.
- **Festival effect:** `ramp × strength` predicts it far better than either analog.
  - Fresh: r = 0.92 for ramp × strength, against 0.71 for the ISO analog and 0.06 for the festival-relative analog.
  - The festival-relative analog fails because the festival falls on a different weekday each year.
- **Chilled share:** last year's share for the same week is the best feature, with MAE 0.45 pp.

---

## Phase 5: Modelling, from simple to more complex

Build these in order. Each one is the baseline the next must beat.

### 5.1 Baselines (about 1 hour; the safety net)

- **Seasonal naive:** the same week last year × the growth ratio, where the growth ratio is the last 8 weeks this year divided by the same 8 weeks last year.
- **Recent‑mean naive:** mean of the last 8 normal weeks × `n_operating_days / 6`.

### 5.2 Multiplicative structural model (likely the strongest and easiest to explain)

```
weekly_volume = base_daily_level(series)
                × Σ over operating days [ dow_effect × payday_effect × festival_effect(ramp, days to festival) ]
```

- Estimate the effects from history at daily level, per brand. Fresh and Style react differently. Tech is mostly noise.
- `base_daily_level` = recent de‑seasonalized level. *(Updated after Phase 4: no trend term. A trend extrapolated over the 10 weeks made every brand worse. Use the mean of the last 8–13 clean weeks.)* *(Phase 6: a year-on-year growth correction was adopted for Fresh and Style; see Phase 6.)*
- Festival effect = `ramp × (fest_strength − 1)`, with the strength taken from the last occurrence of the same festival *(Phase 4: r = 0.92 for Fresh)*.
- Easy to justify to judges and easy to show in the video.

### 5.3 Global ML model (LightGBM, or Ridge/Poisson GLM given how little data there is)

- Model either the daily panel (about 4,000 rows) or the weekly panel with simulated origins.
- Target: volume, or log volume, or the ratio to the base level. Predicting the **ratio** lets one model learn the festival multipliers across all 6 series.
- Use strong regularization and shallow trees, because there are very few festival examples.

### 5.4 Tech series

- Too noisy for a complex model. Use a shrunk recent mean × operating‑day adjustment, possibly with a mild festival uplift. Don't chase spikes.

### 5.5 Chilled volume

- `pred_chilled = pred_total_fresh × chilled_share`. Model it as a share, not as a separate volume, so chilled can never exceed total.
- *(Updated after Phases 3–4.)* The share follows a smooth yearly cycle and festivals don't move it, so use **last year's share for the same week** (`share_ly_target`).
  - Accuracy (MAE): 0.45 pp for `share_ly_target`, 0.48 pp for `share_seasonal` (with the 2026-vs-2025 offset added), 0.69 pp for the recent 6-week share.
  - The recent share under-forecasts the F1 window by 0.76 pp.
  - Phase 6 confirms `share_ly_target` against `share_seasonal`.
- Style and Tech chilled values are hard‑coded to **0**.

### 5.6 Ensemble

- Blend 5.2 and 5.3 (a simple average or a weighted average chosen on validation). This is usually more robust than either model alone.

### As implemented (updated after Phase 5)

Every model is a function of the origin and uses only data up to it. Forecasts for 96 origins are saved to `data/processed/backtest_preds.parquet` for Phase 6.

**Baselines:**
- `B1_SEASONAL` = `ly_total × yoy8`. The growth ratio uses *clean* weeks, so festivals don't distort it.
- `B2_RECENT` = `lvl8 × n_operating_days`.

**Structural model (daily):**
```
L × f[dow] × (1 + p × payday) × (1 + b[brand, festival] × ramp)
```
The week's forecast is the sum over its operating days.
- `f` (day-of-week factor): learned from clean weeks.
- `p` (payday uplift): learned per brand.
- `L` (level): taken from the last 13 clean weeks.
- `b` (festival slope): for each past festival, least squares on its ramp days relative to the level just before it. The value used is the **mean over that festival's past occurrences** (Phase 4's `fest_strength` used only the last one).
- **Tech:** flat day-of-week factors, a 26-week level, and one festival slope pooled over all festivals.

**LightGBM:**
- Target: `y_total / (lvl13 × dow_weight)`.
- Loss: L1, weighted by `lvl13 × dow_weight`.
- Shallow trees.
- Refit at every origin, using only rows whose *target* week is ≤ the origin.
- Starts at origin 2024 wk 28, the first with at least 600 labelled rows, so there is no LightGBM forecast for F2.
- The Ridge/GLM alternative was not built, because the structural model already fills the role of a simple, explainable model.

**Chilled:** `pred_total × share_ly_target` for every model except `B2_RECENT`, which uses the recent share.

**Ensemble:** a 50/50 average of `STRUCT` and `LGBM` for now. Phase 6 sets the weights **per brand**.

**Results:** WAPE over total and chilled, on 49 origins and on F1.

| Model | All origins | F1 |
|---|---|---|
| Structural | 4.5% | 4.1% |
| LightGBM | 5.1% | 8.2% |
| Ensemble | 4.4% | 5.1% |
| Naive baselines | about 7.5% | about 11.5% |

- **Tech:** every model is at 30–37%.

---

## Phase 6: Validation (the most important phase)

Use the same horizon structure as the real task. **Never use a random split.**

| Fold | Train up to | Forecast | Why it matters |
|---|---|---|---|
| **F1 (primary)** | 2025 week 13 | 2025 weeks 14–23 | **Almost identical to 2026:** New Year on a Monday in week 16, festival build‑up, Vesak in May. |
| F2 | 2024 week 13 | 2024 weeks 14–23 | Only 13 weeks of history, so it tests robustness. |
| F3–F5 | Rolling origins, e.g. 2025 weeks 26, 39 and 52 | The following 10 weeks each | Checks performance on normal, non‑festival periods. |

- Report WAPE and MAE **per series** and **per horizon**, and separately for **festival weeks vs normal weeks**.
- Plot actual vs forecast for F1 for every series. That figure also works well in the video.
- Pick the model or blend weights from F1 plus the average over all folds. Don't tune on F1 alone, because that overfits one festival season.

### As implemented (updated after Phase 6)

**Folds:**
- **F1:** origin 2025 wk 13.
- **F2:** origin 2024 wk 13. Only Recent-mean naive and Structural can run here: there is no last year, and too few rows to train LightGBM.
- **F3, F4, F5:** origins 2025 wk 26, wk 39 and wk 52.
- **All origins:** 49 rolling origins, 2025 wk 7 to 2026 wk 3. This was added as the most stable average.

**Metrics:**
- WAPE on total and chilled together (headline);
- WAPE on total and on chilled separately;
- MAE, RMSE and bias;
- broken down per series, per horizon, and per week type (normal, run-up, short).

**Selection rule:** ½ × F1 + ½ × all origins, per brand.

**Added in this phase: a growth correction.** `STRUCT_G = STRUCT × yoy8 ^ ((h + 6.5) / 52)`, for Fresh and Style.
- Every model forecast too low, by −1.9% for Structural, because the 13-week level lags growth.
- Unlike Phase 4's `trend26`, this uses year-on-year growth of clean weeks, which is stable.
- Effect: F1 4.07 → 3.56, F3 5.50 → 4.96, F5 unchanged, F4 4.12 → 4.70. It was adopted.

**Checked and kept:**
- A 13-week level window. 8 weeks ties overall but is worse on F1 and F2.
- The mean festival slope. "Last" makes no difference, because the slopes are stable year to year.
- `share_ly_target` for chilled.
- **No** Tech wk 17 adjustment. The pattern doesn't hold: Kandy 2025 wk 17 was 0.95.

**Chosen per brand**, saved in `models/task2a/phase6_selection.json`:
- **Fresh:** 100% Structural + growth.
- **Style:** 100% Structural + growth.
- **Tech:** 100% LightGBM.

**Result (FINAL):**
- All origins: WAPE 4.30%, bias −0.6%.
- F1: 3.49%.
- Naive baselines for comparison: 7.5% on all origins and 11.6% on F1.
- F2: every model is at about 10.5–11%, because there is no past festival season to learn from. It is the worst case, not the expected error.

---

## Phase 7: Final fit, prediction and submission

1. Refit the chosen models on **all** data up to 2026 week 13.
2. Predict the 60 rows in `task2a_test_inputs.csv`. Merge on `depot, brand, iso_year, iso_week`.
3. **Checks before writing the file:**
   - There are 60 rows, and the `row_id` order and values exactly match the template.
   - No NaN values, and all predictions are ≥ 0.
   - `pred_chilled_volume_m3 == 0` for Style and Tech, and chilled ≤ total for Fresh.
   - Sanity plot: history plus the forecast for each series. Week 15 (before New Year) should be high, week 16 low, and week 18 low (5 operating days).
   - Compare the totals with the same weeks of 2025 × the growth ratio. Any big difference should have a clear reason.
4. Write `submission_task2a.csv`.

### As implemented (updated after Phase 7)

**Final fit and saved models.** The chosen models were refit on all data up to 2026 wk 13 and saved to `models/task2a/`:

| File | Content |
|---|---|
| `structural_model.json` | All learned parameters of the structural model, human-readable |
| `lgbm_model.txt` | The LightGBM booster, used for Tech |
| `phase6_selection.json` | Which model each brand uses, plus the validation scores |
| `manifest.json` | Training period, the inputs needed at inference, and library versions |

The structural coefficients are saved as JSON rather than pickle, and the LightGBM model as LightGBM's own text format rather than joblib. Both are readable and don't depend on library versions.

**Inference pipeline: the last cell of the notebook (7.4).** It is standalone and uses only files on disk:
- the models in `models/task2a/`;
- the test inputs and the template;
- `features_origin.parquet` (the rows for the 2026 wk 13 origin);
- `calendar.parquet` (the days of wk 14–23).

It runs these steps:
1. Load the models.
2. Build the inputs for the 60 test rows.
3. Structural forecast, day by day.
4. Growth correction.
5. LightGBM forecast.
6. Per-brand choice.
7. Chilled = total × last year's share.
8. Validate and write `submissions/submission_task2a.csv` in template order.
9. Print the inputs and predictions.

It was verified in a fresh Python process (identical output), and it reproduces the validated Phase 6 forecast exactly.

**Result:**
- Total 18,685 m³, of which 5,956 m³ chilled.
- Fresh: +5.6% to +7.7% vs 2025, matching its own year-on-year growth (−0.3% to −0.8% vs 2025 × growth).
- Tech: +15% to +34% vs 2025. Tech really did grow; Peliyagoda-Tech in 2026 wk 1–13 was +28% vs 2025.
- Sanity plot: `p7_01_final_forecast.png`.

---

## Phase 8: Deliverables for Task 2A

- **Model files:** save to `models/task2a/` (joblib or pickle for the ML model, JSON or pickle for the structural model's coefficients, and the chilled‑share table).
- **Notebook inference cell:** load the saved models, build the features for the test rows, and **print the inputs and predictions**. The booklet requires this explicitly.
- **Preprocessing document, Task 2A section:** cover why weeks are assigned by `order_date`, why deferred and not_run orders are included, how train and test inputs are merged, the missing `not_run` orders in the 2026 test period, the date‑format fix, how features are built without leakage, and the chilled‑as‑share decision.
- **Architecture diagram, Task 2A part:** raw files → order union → calendar join → daily and weekly panels → features → structural + ML models → ensemble → chilled split → submission. Also show deployment: a weekly batch retrain each Monday.
- **Video, about 1 minute for 2A:** the festival event‑study plot, the F1 backtest plot, and the final forecast plot.
- **AI tool disclosure:** record what was used and where.

---

## Common mistakes to avoid

1. Grouping by `dispatch_date`, or dropping deferred and not_run orders.
2. Leaving out `task1_test_inputs.csv`, which leaves a 6‑week gap right before the forecast.
3. Computing ISO weeks manually instead of using `calendar.csv`, or misreading its `M/D/YYYY` date format.
4. Using lag‑1 or lag‑2 features that won't exist for 10‑week‑ahead predictions.
5. Ignoring operating days: week 16 has 4 operating days and week 18 has 5.
6. Treating the festival week as the peak. The surge comes *before* it, and the festival week itself drops.
7. Fitting a large model to 117 weeks × 6 series. Overfitting the festival weeks is the main risk.
8. Chilled > total, or a non‑zero chilled value for Style or Tech.
9. Changing the `row_id` order or values in the submission.

---

## Schedule (deadline: Fri 9 Oct 2026, 11:59 PM Sri Lanka time)

- **Wed 7 Oct:** Phases 1–3. Get the baseline (5.1) submitted as a safe first version.
- **Thu 8 Oct, morning:** structural model (5.2), validation folds (Phase 6), and the chilled share.
- **Thu 8 Oct, afternoon:** LightGBM or GLM (5.3) and the ensemble, only if time allows and only if it beats 5.2 on F1.
- **Fri 9 Oct:** final fit, submission checks, model files, inference cell, the 2A sections of the document and diagram, and the video clip.
