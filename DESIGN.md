# Retirement & Tax Strategy Model — Design Document

**Status:** v0 built (deterministic, federal tax only). Update the Decisions Log and Roadmap at the end of every work session.
**Privacy rule:** Real personal data lives only in `config.local.toml` on the user's machine. Never paste real balances or birth dates into chat; use the template.

---

## 1. Purpose

Build a numerical model to compare retirement withdrawal and tax strategies (withdrawal order, Roth conversions, Social Security timing, inheritance handling, state moves) so the user can find good paths and avoid terrible ones. Build small, add one feature at a time, never rewrite the core.

## 2. Evaluation Metrics

| # | Metric | Definition |
|---|---|---|
| 1 | **W** | Maximum constant monthly, real (start-year dollars), after-tax household spending such that the portfolio reaches ~$0 at the horizon (95th birthday of the horizon person). Found by bisection. |
| 2 | **Flat-spending balance path** | Account balances (real and nominal) over time at a fixed target spending level. |
| 3 | **Curve-spending balance path** | Same, but spending follows an age-based curve (`spending.curve`, multiplier on the target). |

Planned additional metrics: probability of success and W at 90% success (Monte Carlo); 10th-percentile outcomes; W if one spouse dies early; W without inheritance; after-tax estate at the horizon and a minimum-bequest constraint; lifetime federal/state tax; lifetime IRMAA surcharges; peak marginal bracket reached; minimax regret across inheritance timings; impact of each dated purchase on W.

**Known caveats of W (keep in mind when reading results):**
- A fixed death date at 95 favors delaying Social Security and Roth conversions; test other death scenarios.
- W treats leftover money as worthless; pair it with an after-tax estate metric once inheritances exist.
- A deterministic W hides sequence-of-returns risk; use Monte Carlo.
- Hold asset allocation constant when comparing tax strategies, or allocation will swamp tax effects.
- Small differences (1–3%) are within the noise of assumptions; use W to eliminate bad strategies, not to rank near-ties.
- Optimizers will park income $1 under an IRMAA cliff; add safety margins or robustness tests.

## 3. Architecture

Three layers plus wrappers, so new features are additions, not rewrites:

1. **Inputs / assumptions** — `config.local.toml` → `Params` (`model.load_params`). Tax law lives in data tables (`taxes.py`).
2. **Year-by-year engine** — `model.simulate()`. Taxes, withdrawals, growth for each year.
3. **Strategy layer** — rules for withdrawal order, conversions, spending. v0: `withdrawal_order` only.
4. **Solver** — `model.solve_max_w()` (bisection on W; circular tax gross-up is solved inside each year by bisection on the gross draw).
5. **Wrappers (planned)** — scenario grids (inheritance timing), Monte Carlo (returns, inflation, death dates).

**Files:** `DESIGN.md`, `config.template.toml`, `taxes.py`, `model.py`, `run.py`, `test_model.py`, `.gitignore`.
**Run:** `python run.py` (reads `config.local.toml`). **Test:** `python -m unittest -v`. Python 3.11+, standard library only.

## 4. Conventions

- Annual time step in v0 (monthly later if needed). Withdrawals at start of year, growth over the year.
- **Real dollars = dollars of `start_year`.** Nominal = real × (1+inflation)^k. Track both.
- Tax tables are stored in `BASE_YEAR` (2026) dollars and indexed by inflation via a `scale` factor.
- W is monthly, real, **after-tax** spending. Taxes are computed inside the engine.
- Subscripts: 1 = you, 2 = spouse, F = father, M = mother; t = month, y = year, i = Monte Carlo path.
- Success = no unfunded spending (tolerance $0.01).
- Verify all tax figures against current IRS publications before trusting results.

## 5. Variable List

No pension variables (household has no pensions).

### A. Timeline and people
| Symbol | Name | Description |
|---|---|---|
| `t₀` | Model start date | First period simulated |
| `B₁`, `B₂` | Birth dates | Drive ages and age-based rules |
| `a_end` | Target end age | Age at which principal reaches 0 (95) |
| `T_end` | Horizon end date | 95th birthday of the horizon person |
| `D₁`, `D₂` | Death dates | Random or scenario-driven |
| `S(a)` | Survival probability | Mortality table for random deaths |
| `D_mode` | Death scenario flag | Fixed 95, early death, or random |
| `R₁`, `R₂` | Retirement dates | When each spouse stops working |
| `SS₁`, `SS₂` | Social Security claim dates | When each starts benefits |
| `M₁`, `M₂` | Medicare start dates | Normally 65th birthday |
| `Age₁(t)`, `Age₂(t)` | Current ages | Derived |
| `F(t)` | Filing status | Joint until first death, then single |
| `T_move` | State move date | Relocation date, if any |
| `St_from`, `St_to` | States of residence | Before/after move |

### B. Starting balances and assets
| Symbol | Name | Description |
|---|---|---|
| `P_trad,1`, `P_trad,2` | Traditional IRA/401(k) balances | Pre-tax, per spouse |
| `P_roth,1`, `P_roth,2` | Roth balances | Tax-free, per spouse |
| `P_tax` | Taxable brokerage balance | Market value |
| `C_tax` | Taxable cost basis | For capital gains |
| `P_cash` | Cash and savings | Non-invested reserves |
| `P_hsa` | HSA balance | Triple-tax-advantaged medical account |
| `Hold(j)` | Holdings detail | Per-fund/stock asset class and weights (later) |
| `Ann` | Annuity income | Other annuity payments, terms |
| `H_val`, `H_g` | Home value and appreciation | Market value and growth |
| `Mtg` | Mortgage | Balance and payment |
| `A_other` | Other assets | Rental property, business interests |

### C. Income before and during retirement
| Symbol | Name | Description |
|---|---|---|
| `Sal₁`, `Sal₂` | Salaries | Pre-retirement earned income |
| `g_sal` | Wage growth | Annual salary growth |
| `PT₁`, `PT₂` | Part-time income | Post-retirement earnings and end date |
| `Rent` | Rental/other income | Recurring income |
| `Contrib` | Retirement contributions | Pre-retirement saving by account type |
| `CU` | Catch-up contributions | Includes Roth-only rule for high earners |

### D. Social Security
| Symbol | Name | Description |
|---|---|---|
| `PIA₁`, `PIA₂` | Primary insurance amounts | Benefit at full retirement age |
| `FRA₁`, `FRA₂` | Full retirement ages | Based on birth year |
| `k_claim` | Claiming adjustment | Reduction/credit for early/delayed claiming |
| `COLA` | SS cost-of-living adjustment | Annual increase |
| `SSB(t)` | Monthly SS benefit | After adjustments |
| `Spouse_ben` | Spousal benefit | Based on other spouse's record |
| `Surv_ben` | Survivor benefit | Survivor keeps the larger check |
| `δ_SS` | Benefit cut factor | Possible trust-fund shortfall reduction |
| `SS_tax` | Taxable SS fraction | 0–85% by provisional income |
| `PI` | Provisional income | AGI + tax-exempt interest + half of SS |
| `PI_thr` | Provisional income thresholds | Unindexed tiers by filing status |

### E. Inheritance and inherited accounts
Working assumption: each event ≈ $1.5M (range $1.0M–$2.0M real), majority (~80%) in a traditional IRA. Timing is unknown: handle with a scenario grid, minimax regret, contingent rules, and Monte Carlo timing.

| Symbol | Name | Description |
|---|---|---|
| `Age_F`, `Age_M` | Parents' current ages | For mortality-based timing |
| `Mort_par` | Parents' mortality table | Conditioned on current age |
| `D_F`, `D_M` | Parents' death dates | Random or scenario grid |
| `Inh_grid` | Inheritance timing grid | Candidate dates for each event |
| `Inh_F`, `Inh_M` | Inheritance amounts | Real dollars, with ranges |
| `Mix_F`, `Mix_M` | Asset mix of inheritance | IRA / Roth / taxable / cash / real estate split |
| `Step` | Basis step-up | Resets basis on inherited taxable assets |
| `P_inhIRA` | Inherited IRA balance | Pre-tax balance received |
| `Roth_inh` | Inherited Roth balance | 10-year rule, no annual RMD |
| `RBD_flag` | Parent past RMD start flag | If yes, annual RMDs in years 1–9 |
| `RMD_dec` | Decedent's final-year RMD | Owed for year of death |
| `T_10` | 10-year deadline | Account must be empty by then |
| `RMD_inh(t)` | Inherited IRA RMD | Single-life table based |
| `W_inh(t)` | Inherited withdrawal schedule | Chosen draws within the rules |
| `Overlap` | Window overlap flag | Whether the two 10-year windows overlap |
| `Sib_share` | Heirs' share | Your fraction if other heirs exist |
| `Est_cost` | Estate costs and delays | Legal fees, probate timing |
| `Est_tax_St` | State estate/inheritance tax | On parents' estate, if applicable |

### F. Spending and liabilities
| Symbol | Name | Description |
|---|---|---|
| `W` | Monthly spending (real, after-tax) | Headline metric |
| `s(t)` | Spending shape | Age multiplier (flat or curve) |
| `W_ess`, `W_disc` | Essential/discretionary spending | Flexibility split |
| `W_floor` | Spending floor | Minimum acceptable spending |
| `Heal_med` | Medicare costs | Part B, D, Medigap, out-of-pocket |
| `Heal_pre65` | Pre-Medicare coverage cost | ACA or bridge premiums |
| `g_heal` | Healthcare inflation | Above general inflation |
| `LTC_mode` | LTC approach | Self-insure, insurance, set-aside |
| `LTC_p` | LTC event probability | Chance of needing care |
| `LTC_t`, `LTC_dur` | LTC onset and duration | Start and length |
| `LTC_cost`, `g_LTC` | LTC cost and inflation | Today's dollars; care-specific inflation |
| `LTC_prem`, `LTC_ben` | LTC insurance premium/benefit | If insured |
| `LTC_set` | LTC set-aside | Earmarked reserve |
| `Med_ded` | Deductible medical expenses | Above 7.5% AGI floor |
| `PropTax` | Property tax / housing costs | Annual |
| `T_down`, `Down_net` | Downsizing date / net proceeds | Home sale timing and cash |
| `Give` | Charitable giving | Annual amount |
| `QCD` | Qualified charitable distribution | IRA giving after 70½ |
| `OneTime(t)` | One-time purchases | Cars, vacations, repairs (**v0: implemented**) |

### G. Markets and economy
| Symbol | Name | Description |
|---|---|---|
| `μ_e`, `σ_e` | Equity return, volatility | Mean and std. dev. |
| `μ_b`, `σ_b` | Bond return, volatility | Mean and std. dev. |
| `μ_c` | Cash return | Interest on cash |
| `ρ` | Return correlation | Equities, bonds, inflation |
| `π`, `σ_π` | Inflation and volatility | Mean and variability |
| `Π(t)` | Price index | Cumulative inflation factor |
| `w_eq(t)` | Equity allocation | Glidepath |
| `Rebal` | Rebalancing rule | Frequency/threshold |
| `Fee` | Expense ratio | Annual fund fees |
| `y_div`, `y_int` | Dividend/interest yield | Taxable income from taxable accounts |
| `Gain_real` | Realized gain fraction | Annual tax drag |
| `Ret_model` | Return model | Normal, lognormal, bootstrap, regime |
| `N_sim` | Number of simulations | Monte Carlo paths |
| `Seed` | Random seed | Reproducibility |
| `p_succ` | Success threshold | e.g. 90% |

### H. Tax law parameters (year-indexed data tables)
| Symbol | Name | Description |
|---|---|---|
| `Br(y,F)` | Federal brackets | Rates/thresholds by status |
| `SD(y,F)` | Standard deduction | By year and status |
| `SrDed` | Senior deduction | $6,000 per person 65+, 2025–2028 |
| `SrDed_ph` | Senior deduction phase-out | MAGI threshold (~$150k MFJ) |
| `LTCG(y,F)` | Capital gains brackets | 0/15/20% thresholds |
| `NIIT` | Net investment income tax | 3.8% above threshold |
| `IRMAA(y,F)` | IRMAA tiers/premiums | MAGI cliffs and surcharges |
| `LB` | IRMAA lookback | 2-year delay |
| `Law_flag` | Tax law scenario | Current law vs. higher rates |
| `g_idx` | Bracket indexing rate | Chained CPI |
| `RMD_age` | RMD starting age | 73 now; 75 if born 1960+ |
| `UL(a)`, `JL(a₁,a₂)`, `SL(a)` | RMD tables | Uniform, joint, single-life (inherited) |
| `RMD_pen` | RMD penalty | 25% of missed amount |
| `EWP` | Early withdrawal penalty | 10% before 59½ |
| `ACA(y)` | ACA subsidy schedule | By MAGI before Medicare |
| `St_tax` | State income tax schedule | Rates/brackets by state |
| `St_SS` | State SS taxation | Taxed or not |
| `St_ret` | State retirement exclusions | Pension/IRA/inherited-IRA treatment |

### I. Strategy (decision) variables
| Symbol | Name | Description |
|---|---|---|
| `Order` | Withdrawal sequence | Order of drawing buckets (**v0: implemented**) |
| `Conv(y)` | Roth conversion amount | Dollars converted per year |
| `Conv_start`, `Conv_end` | Conversion window | First/last years |
| `Conv_rule` | Conversion method | Bracket-filling or IRMAA ceiling |
| `Br_tgt` | Target bracket | Top bracket to fill |
| `IRMAA_tgt` | Target IRMAA tier | MAGI ceiling |
| `Margin` | Safety margin | Buffer under cliffs |
| `Conv_src` | Conversion tax source | Cash, brokerage, or the IRA |
| `Harvest`, `Loss` | Gain/loss harvesting rules | Use 0% LTCG bracket; use losses |
| `Cash_tgt` | Cash reserve target | Months of spending |
| `Spend_rule`, `Guard` | Spending rule / guardrails | Fixed, guardrails, percentage |
| `Inh_rule` | Inherited IRA draw strategy | Even, front-load, back-load |
| `Own_acct` | Beneficiary choice | Titling, which spouse inherits |
| `Order_acct` | Spouse account ordering | Whose accounts to draw first |
| `Dis_mode` | Post-first-death plan change | How plan adjusts |

### J. Computed and state variables
| Symbol | Name | Description |
|---|---|---|
| `Bal_a(t)` | Balance by account | Per bucket over time |
| `Wd_a(t)` | Withdrawals by account | Amount drawn per bucket |
| `RMD(t)` | RMD owed | Own RMD |
| `AGI(t)`, `MAGI(t)`, `TI(t)` | Income measures | AGI, modified AGI, taxable income |
| `Tax_fed(t)`, `Tax_st(t)` | Taxes | Federal and state |
| `τ_marg`, `τ_eff` | Marginal/effective rates | Federal + state |
| `IRMAA_tier(t)`, `IRMAA_sur(t)` | IRMAA tier and surcharge | From lookback year |
| `Gap(t)` | Funding shortfall | Unfunded spending |

### K. Outputs and metrics
| Symbol | Name | Description |
|---|---|---|
| `W*` | Base-case maximum W | Depletes principal at `T_end` |
| `W₉₀` | W at 90% success | Monte Carlo |
| `W₁₀` | 10th-percentile W | Low-end outcome |
| `W_early` | W under early death | One spouse dies early |
| `W_noInh` | W without inheritance | Tests dependence |
| `Est(T_end)` | After-tax estate | Net value left to heirs |
| `B_min` | Minimum bequest constraint | Required legacy |
| `Tax_life` | Lifetime taxes | Federal + state |
| `IRMAA_life` | Lifetime IRMAA | Total surcharges |
| `P_succ` | Probability of success | Fraction of paths funded |
| `ε_W` | Solver tolerance | Stopping criterion |

## 6. v0 Scope

**Implemented:** TOML config loader; four buckets (cash, taxable w/ basis, traditional, Roth) combined across spouses; single nominal return and inflation; federal ordinary + LTCG tax with standard deduction, indexed by inflation; fixed withdrawal order; after-tax gross-up; bisection solver for W; flat and curve spending; dated real-dollar purchases; CSV output; unit tests with closed-form checks.

**Known simplifications in v0:**
- One return for all buckets (no dividend/interest tax drag in the taxable account).
- Filing status fixed for the whole horizon (no death or single-filer switch).
- No Social Security, RMDs, IRMAA, senior deduction, NIIT, state tax, or inheritance.
- Spending curve keys off person 1's age.
- Year-based ages; ending balance measured at end of the year before the horizon birthday year.
- Because RMDs are not modeled, "traditional" money is drawn only when the order says so.

## 7. Roadmap (planned build order)

1. Social Security + its taxation (provisional income)
2. RMDs and the first-death / single-filer switch
3. Roth conversions (bracket-filling, then IRMAA ceiling with 2-year lookback) and senior deduction
4. Inheritance events + inherited IRA rules (scenario grid for timing)
5. Healthcare, LTC, spending curve refinements
6. State taxes and the move date
7. Monte Carlo wrapper (returns, inflation, deaths, parent timing)
8. Additional metrics (estate, regret, success probability)

## 8. Decisions Log

| Date | Decision |
|---|---|
| 2026-10-21 | W = constant real, after-tax monthly spending; solved by bisection. |
| 2026-10-21 | Three headline metrics: W, flat-spending balance, curve-spending balance. |
| 2026-10-21 | Personal data stays local in `config.local.toml`; code uses a template with fake numbers. |
| 2026-10-21 | No pensions in the model. |
| 2026-10-21 | Inheritance assumed ~$1.5M each, mostly traditional IRA; timing handled via scenario grid / regret / Monte Carlo, not a single guessed date. |
| 2026-10-21 | v0: annual steps, Python 3.11+ stdlib only (TOML config). |

## 9. Open Questions

- Parents' and own ages (kept private; supply via config only) to set realistic inheritance-timing scenarios.
- Whether legacy for children matters (decides whether to add a minimum-bequest constraint early).
- Verify 2026 tax tables (standard deduction sources conflicted: $30,000 vs $32,200 for MFJ; v0 uses $32,200).
