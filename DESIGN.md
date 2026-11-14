# Retirement & Tax Strategy Model — Design Document

**Status:** v8 built (spousal Social Security benefit; asset classes with dividends and interest taxed each year; v7: inheritances from the parents, with the inherited-IRA 10-year rules; v6: health insurance premiums: marketplace before 65, Medicare plus IRMAA after; v5: state income tax with an optional move, and a generated data dictionary; v4: required minimum distributions; v3: death dates, single-filer switch, survivor Social Security; deterministic; federal tax incl. Social Security taxation; per-person accounts; 59½ rule; selectable withdrawal strategies; per-person Social Security claim age 62–70 with early/delayed adjustments). Update the Decisions Log and Roadmap at the end of every work session.
**Privacy rule:** Real personal data lives only in `config.local.toml` on the user's machine. Never paste real balances or birth dates into chat; use the template.

---

## 1. Purpose

Build a numerical model to compare retirement withdrawal and tax strategies (withdrawal order, Roth conversions, Social Security timing, inheritance handling, state moves) so the user can find good paths and avoid terrible ones. Build small, add one feature at a time, never rewrite the core.

## 2. Evaluation Metrics

| # | Metric | Definition |
|---|---|---|
| 1 | **W** | Maximum constant monthly, real (start-year dollars), after-tax household spending such that the portfolio reaches ~$0 at the horizon (the last death; by default the 95th birthday of the younger person). Found by bisection. |
| 2 | **Flat-spending balance path** | Account balances (real and nominal) over time at a fixed target spending level. |
| 3 | **Curve-spending balance path** | Same, but spending follows an age-based curve (`spending.curve`, multiplier on the target). |

Planned additional metrics: probability of success and W at 90% success (Monte Carlo); 10th-percentile outcomes; W if one spouse dies early; W without inheritance; after-tax estate at the horizon and a minimum-bequest constraint; lifetime federal/state tax; lifetime IRMAA surcharges; peak marginal bracket reached; minimax regret across inheritance timings; impact of each dated purchase on W.

**Known caveats of W (keep in mind when reading results):**
- A fixed death date at 95 favors delaying Social Security and Roth conversions; test other death scenarios. Death dates are now inputs (v3) but default to 95, so the bias remains until you (or Monte Carlo) vary them.
- W treats leftover money as worthless; pair it with an after-tax estate metric once inheritances exist.
- A deterministic W hides sequence-of-returns risk; use Monte Carlo.
- Hold asset allocation constant when comparing tax strategies, or allocation will swamp tax effects.
- Small differences (1–3%) are within the noise of assumptions; use W to eliminate bad strategies, not to rank near-ties.
- Optimizers will park income $1 under an IRMAA cliff; add safety margins or robustness tests.

## 3. Architecture

Three layers plus wrappers, so new features are additions, not rewrites:

1. **Inputs / assumptions** — `config.local.toml` → `Params` (`model.load_params`). Tax law lives in data tables (`taxes.py`).
2. **Year-by-year engine** — `model.simulate()`. Taxes, withdrawals, growth for each year.
3. **Strategy layer** — rules for withdrawal order, conversions, spending. v1: named withdrawal plans in `model.STRATEGIES` (an ordered list of `(category, cap)` steps; `cap` = none or the indexed standard deduction). Roth conversions come later.
4. **Solver** — `model.solve_max_w()` (bisection on W; circular tax gross-up is solved inside each year by bisection on the gross draw).
5. **Wrappers (planned)** — scenario grids (inheritance timing), Monte Carlo (returns, inflation, death dates).

**Files:** `DESIGN.md`, `config.template.toml`, `taxes.py` (tax tables and functions), `statetax.py` (state income tax tables and function), `health.py` (Medicare Part B/D premium and IRMAA tables), `rules.py` (SS full retirement age/timing, 59½ rule, RMD table), `columns.py` (documentation of every CSV column), `model.py`, `run.py`, `test_model.py`, `golden/representative_scenario.json` (regression baseline), `.gitignore`.
**Run:** `python run.py` (reads `config.local.toml`). **Test:** `python -m unittest -v`. Python 3.11+, standard library only.

## 4. Conventions

- Annual time step in v0 (monthly later if needed). Withdrawals at start of year, growth over the year.
- **Real dollars = dollars of `start_year`.** Nominal = real × (1+inflation)^k. Track both.
- Tax tables are stored in `BASE_YEAR` (2026) dollars and indexed by inflation via a `scale` factor.
- W is monthly, real, **after-tax** spending. Taxes are computed inside the engine.
- Subscripts: 1 = you, 2 = spouse, F = father, M = mother; t = month, y = year, i = Monte Carlo path.
- Success = no unfunded spending (tolerance $0.01).
- Verify all tax figures against current IRS publications before trusting results.
- **Social Security:** the config value is the monthly benefit at Full Retirement Age (FRA) in start-year dollars, multiplied by a claim-age factor; it stays constant in real terms (COLA = inflation). Each person's claim age is `[years, months]` (default: FRA, allowed 62y0m–70y0m). Benefits start in the month containing the claim-age birthday and are prorated by month in the first year (e.g. an April birthday → 9 months that year). Surplus income (benefits exceeding spending plus tax) is saved to cash.
- **Claim-age factor:** before FRA, −5/9 of 1% per month for the first 36 months early, −5/12 of 1% per month beyond 36 (FRA 67: 70% at 62, 75% at 63, 80% at 64, 86.7% at 65, 93.3% at 66). After FRA, +2/3 of 1% per month up to age 70 (124% at 70 for FRA 67). The 2/3% credit is the rule for people born 1943+.
- **Units in outputs:** every dollar column is nominal unless its name ends in `_real`. `price_index` converts any nominal column to start-year dollars (`nominal / price_index`); end-of-year balances use `price_index × (1 + inflation)`. The console table shows Spend(real) and Spend(nom) side by side so Social Security can be compared with nominal spending.
- **Provisional-income thresholds ($32k/$44k MFJ, $25k/$34k single) are NOT inflation-indexed**; they are compared against nominal income, so more benefit becomes taxable over time.
- **Deaths (v3):** each person has a death date (`person1_death`, `person2_death` in `[household]`; default = their `horizon_age` birthday). The model runs to the **last death**: the final simulated year is the year *before* the last death year (the survivor's final year is not simulated). The household files **jointly through the year of the first death** (IRS rule) and **single from the next year**. A one-person household never switches.
- **Survivor Social Security (v3):** from the month of the first death (couples), the household receives ONE check: the larger of the survivor's own benefit (once claimed) and the survivor benefit, taken month by month so the survivor can switch to the larger one. Survivor benefit = deceased's PIA × a basis factor: their own claim-adjusted benefit if they had claimed (floor 82.5% of PIA), 100% of PIA if they died before claiming and before FRA, or PIA plus delayed credits earned to death if they died after FRA before claiming. It starts at the death month but not before the survivor is 60, reduced linearly up to 28.5% if taken before the survivor's FRA. Each person's own benefit stops the month before their death month. After the second death: nothing.
- **Accounts at death (v3):** the deceased's traditional and Roth balances remain in their buckets but are treated as the survivor's and are no longer locked by the 59½ rule from the year after death (spousal-beneficiary treatment: penalty-free).
- **Required minimum distributions (v4):** each person must withdraw, from their OWN traditional balance, `prior-Dec-31 balance ÷ IRS Uniform Lifetime divisor for the age reached that year` (the model's start-of-year balance is the prior Dec 31 balance). Start age: 73 if born 1951–1959, 75 if born 1960+. The first RMD is taken in the year the start age is reached (the optional April-1 delay is not modeled). RMDs are taken FIRST and taxed as ordinary income; the withdrawal plan then covers any remaining spending need; an RMD larger than the need is saved to cash (`surplus_to_cash`). Finite plan caps (the `"std"` standard-deduction step) count RMD draws, so an RMD above one standard deduction means no further discretionary traditional draw. Roth balances (IRA or 401(k)) have no RMDs. Spouses cannot combine accounts; each person's traditional IRAs and 401(k)s are one balance (IRAs aggregate; separate 401(k)s must each pay their own RMD but the total is the same).
- **RMDs and death (v4):** in the year of death the deceased's RMD is still due from their own account; afterwards the survivor is assumed to roll the deceased's account into their own, so the survivor's RMD uses the combined balance and the survivor's age (taken from the survivor's own account first).
- **Spousal benefit (v8):** while both spouses are alive, each person whose own full-retirement-age benefit (PIA) is below 50% of the other's also receives a top-up of `0.5 × other's PIA − own PIA`. It is based on the PIA, never on the other's delayed check. It begins when BOTH have claimed (the other spouse must have filed; "deemed filing" ties it to your own claim), so a higher earner who delays also delays it; it is reduced 25/36 of 1% per month for the first 36 months, then 5/12 of 1% per month, that it starts before the recipient's own full retirement age (65% at 62 when FRA is 67), measured at the month it BEGINS; there is no credit for waiting. It ends at the first death (the survivor benefit then applies). Works for any claim ages 62–70 for either person, so scenario grids over both claim ages are supported. `[social_security] spousal_benefit = false` switches it off to see its value.
- **Asset classes (v8):** optional `[[asset_class]]` entries (total return, income yield, qualified share) plus an `[allocation]` giving each account group (cash, taxable, traditional, roth, inherited) a mix that sums to 1. An account's return is the allocation-weighted class return, assuming constant-mix rebalancing (tax-free); its income yield is weighted the same way. In the CASH and TAXABLE accounts the income is earned on the balance left after the year's withdrawals, taxed that year (ordinary part as ordinary income, qualified part at long-term-gains rates, stacked with everything else), reinvested, and added to the taxable account's cost basis. It also counts toward AGI, state tax, provisional income and the IRMAA lookback. Traditional, Roth and inherited accounts are not taxed yearly. With no classes defined the older single `nominal_return` applies to every account with nothing taxed until sold. The annual tax is part of the cost the solver sees, so W is after it.
- **Inheritances (v7):** configured in `[inheritance]`, `[inheritance.father]`, `[inheritance.mother]` (omit = none). Received by person 1. Each parent: `death_date` (**the key unknown**), `birth_date` (needed for the parent's own year-of-death RMD), `share` (your fraction), `ira_value`, and a list of `[[...assets]]` (label, value, `sale_months_after_death`, `sale_cost_pct`). Amounts are in today's dollars and assumed to keep pace with inflation until received (a parent's IRA that shrinks in real terms should be entered smaller). An inheritance is ignored if person 1 dies before the parent.
- **Inherited IRA rules (v7):** the IRA is deposited at the START of the year of death into its own bucket (`inh_f` / `inh_m`). Because the parent was past their RMD start: year of death D = the parent's own RMD (Uniform table at the parent's age; assumed not yet taken); years D+1..D+9 = start-of-year balance ÷ (IRS Single Life factor at person 1's age in D+1, minus 1 for each later year); year D+10 = the whole balance (deadline Dec 31). Policy `"even"` also takes at least balance ÷ years-left each year (level draws); `"min"` takes only the minimums and leaves a lump in year 10. Forced amounts are taken first (like RMDs), count against finite caps such as the `"std"` step, are ordinary income, and never carry the 10% penalty. The `"trad"` category now draws inherited IRAs before your own traditional accounts. After the heir's death the schedule simply continues (successor beneficiary).
- **Inherited assets (v7):** each asset is received in the calendar year of (death month + `sale_months_after_death`) as value × share × inflation factor × (1 − selling cost) and deposited into the **taxable** bucket at full cost basis (inherited property gets a stepped-up basis, so selling soon after death creates no gain; appreciation after death is not modeled). A house you keep and live in is simply left out.
- **Health insurance (v6):** configured in `[healthcare]` (omit the block = no health costs, the pre-v6 behavior). Premiums are an EXTRA required expense paid from after-tax money: `need_nominal = spend_nominal + health_total`, and **W (monthly living spending) excludes them**. Before Medicare each person pays `marketplace_monthly_per_person` (start-year dollars; $1,500 is a placeholder, treated as the full price with no premium tax credit). Medicare starts in the month of each person's 65th birthday: standard Part B premium ($202.90/month in 2026 dollars, built in) plus optional `medicare_other_monthly_per_person` (Medigap, Part D plan) plus IRMAA. Coverage is month-exact and stops the month before death. Amounts grow with inflation plus `premium_growth_above_inflation`; the CMS table amounts are 2026 dollars, your inputs are start-year dollars.
- **IRMAA (v6):** a surcharge on Part B AND Part D, paid by each person on Medicare. Tier = number of thresholds that the AGI from TWO YEARS EARLIER is above (modeled AGI; `prior_year_magi` for the first two model years), compared with the thresholds for the **current year's** filing status. So after a first death the survivor's single thresholds (half the joint ones) are compared with the couple's old joint income for two years (the widow's trap); the SSA-44 appeal is not modeled (conservative). Thresholds are indexed with inflation except the top one ($500k single / $750k joint), which is fixed. Surcharges are charged in the year, at the start-of-year convention.
- **State income tax (v5):** configured in `[state]`: `residence` (e.g. "IN"), optional `move_date` + `move_state` (give both or neither; no date = no move), optional `local_tax_rate`. No `[state]` block = no state tax (the pre-v5 behavior). The state tax is added to the cost the solver sees (`net = draws + Social Security − federal tax − state tax − penalty`), so **W is after ALL income taxes**. Base = modeled federal AGI minus the taxable part of Social Security (if the state exempts it) minus exemptions; flat rate by year from `statetax.STATES`. In the move year the tax is prorated by months (the move month counts as the new state); the local rate applies only in the starting state. Exemptions count the people alive at the start of the year (so they drop after the first death). `tax_enabled = false` also turns state tax off.
- **Data dictionary (v5):** every run writes `data_dictionary.txt` next to the CSVs, built from `columns.COLUMN_DOCS` (what each column is and how it is calculated, with this run's settings on top). The test suite requires the documented columns to equal the CSV columns in the same order, so adding a column without documenting it fails the tests.
- **59½ rule:** an owner's retirement accounts (traditional and Roth) are locked until the calendar year they turn 59½ (if that is by June 30; otherwise the following year). Locked money is touched only after every other source is exhausted, and then a 10% penalty applies and is reported (`penalty_early_withdrawal`, `early_withdrawal_flag`).
- Within the "trad"/"roth" categories, the older person's accounts are drawn first.

## 5. Variable List

No pension variables (household has no pensions).

### A. Timeline and people
| Symbol | Name | Description |
|---|---|---|
| `t₀` | Model start date | First period simulated |
| `B₁`, `B₂` | Birth dates | Drive ages and age-based rules |
| `a_end` | Target end age | Age at which principal reaches 0 (95) |
| `T_end` | Horizon end date | The last death date (v3); default 95th birthday of the younger person |
| `D₁`, `D₂` | Death dates | Random or scenario-driven (**v3: implemented** as config dates; Monte Carlo later) |
| `S(a)` | Survival probability | Mortality table for random deaths |
| `D_mode` | Death scenario flag | Fixed 95, early death, or random |
| `R₁`, `R₂` | Retirement dates | When each spouse stops working |
| `SS₁`, `SS₂` | Social Security claim dates | When each starts benefits |
| `M₁`, `M₂` | Medicare start dates | Normally 65th birthday (**v6: implemented**, month-exact) |
| `Age₁(t)`, `Age₂(t)` | Current ages | Derived |
| `F(t)` | Filing status | Joint until first death, then single (**v3: implemented**) |
| `T_move` | State move date | Relocation date, if any (**v5: implemented**; `move_date` in `[state]`) |
| `St_from`, `St_to` | States of residence | Before/after move (**v5**: `residence`, `move_state`) |

### B. Starting balances and assets
| Symbol | Name | Description |
|---|---|---|
| `P_trad,1`, `P_trad,2` | Traditional IRA/401(k) balances | Pre-tax, per spouse |
| `P_roth,1`, `P_roth,2` | Roth balances | Tax-free, per spouse |
| `P_tax` | Taxable brokerage balance | Market value |
| `C_tax` | Taxable cost basis | For capital gains |
| `P_cash` | Cash and savings | Non-invested reserves |
| `P_hsa` | HSA balance | Triple-tax-advantaged medical account |
| `Hold(j)` | Holdings detail | Per-fund/stock asset class and weights (**v8**: asset CLASSES with a mix per account; individual holdings and tax lots later) |
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
| `k_claim` | Claiming adjustment | Reduction/credit for early/delayed claiming (**v2: implemented**) |
| `SS_claim₁`, `SS_claim₂` | Claim age (months) | Age at which each person claims, 62y0m–70y0m (**v2**) |
| `COLA` | SS cost-of-living adjustment | Annual increase |
| `SSB(t)` | Monthly SS benefit | After adjustments |
| `Spouse_ben` | Spousal benefit | Based on other spouse's record (**v8: implemented**; column `ss_spousal`) |
| `Surv_ben` | Survivor benefit | Survivor keeps the larger check (**v3: implemented**; basis, widow floor, age-60 gate, early reduction) |
| `δ_SS` | Benefit cut factor | Possible trust-fund shortfall reduction |
| `SS_tax` | Taxable SS fraction | 0–85% by provisional income |
| `PI` | Provisional income | AGI + tax-exempt interest + half of SS |
| `PI_thr` | Provisional income thresholds | Unindexed tiers by filing status |

### E. Inheritance and inherited accounts
Working assumption: each event ≈ $1.5M (range $1.0M–$2.0M real), majority (~80%) in a traditional IRA. Timing is unknown: handle with a scenario grid, minimax regret, contingent rules, and Monte Carlo timing.

| Symbol | Name | Description |
|---|---|---|
| `Age_F`, `Age_M` | Parents' current ages | For mortality-based timing (**v7**: parents' `birth_date` is an input, used for the year-of-death RMD; mortality timing is Monte Carlo, later) |
| `Mort_par` | Parents' mortality table | Conditioned on current age |
| `D_F`, `D_M` | Parents' death dates | Random or scenario grid (**v7: implemented** as `death_date` in `[inheritance.father]` / `[inheritance.mother]`; grid and Monte Carlo later) |
| `Inh_grid` | Inheritance timing grid | Candidate dates for each event |
| `Inh_F`, `Inh_M` | Inheritance amounts | Real dollars, with ranges (**v7**: `ira_value` plus a list of assets, today's dollars, assumed to keep pace with inflation) |
| `Mix_F`, `Mix_M` | Asset mix of inheritance | IRA / Roth / taxable / cash / real estate split (**v7**: IRA + a list of assets (houses, land, belongings, cash), each with a sale delay and selling cost; inherited Roth not modeled) |
| `Step` | Basis step-up | Resets basis on inherited taxable assets (**v7**: sale proceeds enter the taxable bucket at full basis, so the sale has no gain) |
| `P_inhIRA` | Inherited IRA balance | Pre-tax balance received (**v7**: buckets `inh_f`, `inh_m`) |
| `Roth_inh` | Inherited Roth balance | 10-year rule, no annual RMD |
| `RBD_flag` | Parent past RMD start flag | If yes, annual RMDs in years 1–9 (**v7**: always assumed yes; enforced by requiring the parent to be at least RMD age at death) |
| `RMD_dec` | Decedent's final-year RMD | Owed for year of death (**v4**: spouses' case; **v7**: parents' inheritance, assumed not yet taken) |
| `T_10` | 10-year deadline | Account must be empty by then (**v7**: Dec 31 of the 10th year after death) |
| `RMD_inh(t)` | Inherited IRA RMD | Single-life table based (**v7: implemented**; column `inherited_ira_required`) |
| `W_inh(t)` | Inherited withdrawal schedule | Chosen draws within the rules (**v7**: `ira_withdrawal` = "even" or "min"; column `draw_inherited_ira`) |
| `Overlap` | Window overlap flag | Whether the two 10-year windows overlap |
| `Sib_share` | Heirs' share | Your fraction if other heirs exist (**v7**: `share`) |
| `Est_cost` | Estate costs and delays | Legal fees, probate timing (**v7**: per-asset `sale_cost_pct` and `sale_months_after_death`) |
| `Est_tax_St` | State estate/inheritance tax | On parents' estate, if applicable |

### F. Spending and liabilities
| Symbol | Name | Description |
|---|---|---|
| `W` | Monthly spending (real, after-tax) | Headline metric |
| `s(t)` | Spending shape | Age multiplier (flat or curve) |
| `W_ess`, `W_disc` | Essential/discretionary spending | Flexibility split |
| `W_floor` | Spending floor | Minimum acceptable spending |
| `Heal_med` | Medicare costs | Part B, D, Medigap, out-of-pocket (**v6**: Part B standard premium built in; Medigap/Part D plan via `medicare_other_monthly_per_person`; out-of-pocket costs not modeled) |
| `Heal_pre65` | Pre-Medicare coverage cost | ACA or bridge premiums (**v6: implemented**; `marketplace_monthly_per_person`, no subsidy model) |
| `g_heal` | Healthcare inflation | Above general inflation (**v6**: `premium_growth_above_inflation`, default 0) |
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
| `μ_c` | Cash return | Interest on cash (**v8**: the "cash" asset class; its interest is taxed yearly) |
| `ρ` | Return correlation | Equities, bonds, inflation |
| `π`, `σ_π` | Inflation and volatility | Mean and variability |
| `Π(t)` | Price index | Cumulative inflation factor |
| `w_eq(t)` | Equity allocation | Glidepath |
| `Rebal` | Rebalancing rule | Frequency/threshold |
| `Fee` | Expense ratio | Annual fund fees |
| `y_div`, `y_int` | Dividend/interest yield | Taxable income from taxable accounts (**v8: implemented** as each class's `income_yield` and `qualified_share`) |
| `Gain_real` | Realized gain fraction | Annual tax drag (**v8**: handled through yearly taxed income; capital-gain turnover inside funds is not modeled) |
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
| `IRMAA(y,F)` | IRMAA tiers/premiums | MAGI cliffs and surcharges (**v6: implemented**, 2026 CMS amounts; thresholds indexed except the top one) |
| `LB` | IRMAA lookback | 2-year delay (**v6: implemented**; `prior_year_magi` supplies the two years before the model starts) |
| `Law_flag` | Tax law scenario | Current law vs. higher rates |
| `g_idx` | Bracket indexing rate | Chained CPI |
| `RMD_age` | RMD starting age | 73 if born 1951–59; 75 if born 1960+ (**v4: implemented**) |
| `UL(a)`, `JL(a₁,a₂)`, `SL(a)` | RMD tables | Uniform (**v4: implemented**), joint (spouse >10 years younger; not yet), single-life (inherited; later) |
| `RMD_pen` | RMD penalty | 25% of missed amount (never incurred: the model always takes the RMD) |
| `EWP` | Early withdrawal penalty | 10% before 59½ |
| `ACA(y)` | ACA subsidy schedule | By MAGI before Medicare |
| `St_tax` | State income tax schedule | Rates/brackets by state (**v5: implemented** for IN, TX, FL as flat rates by year) |
| `Loc_tax` | Local income tax rate | e.g. an Indiana county income tax; added to the starting state's rate, stops at the move (**v5**; `local_tax_rate`) |
| `St_SS` | State SS taxation | Taxed or not (**v5**: `exempt_social_security`) |
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
| `Strat` | Withdrawal strategy name | Which built-in plan is used (**v1: implemented**) |
| `Std_cap` | Standard-deduction cap | Cap on a step's draw, indexed (**v1**) |
| `Early_order` | Last-resort order for locked accounts | Roth vs. traditional when penalized money is unavoidable (**v1**) |
| `SSB_mult` | SS benefit multiplier | Sensitivity knob for a benefit cut (**v1**) |

### J. Computed and state variables
| Symbol | Name | Description |
|---|---|---|
| `Bal_a(t)` | Balance by account | Per bucket over time |
| `Wd_a(t)` | Withdrawals by account | Amount drawn per bucket |
| `RMD(t)` | RMD owed | Own RMD (**v4: implemented**; column `rmd_required`) |
| `AGI(t)`, `MAGI(t)`, `TI(t)` | Income measures | AGI, modified AGI, taxable income |
| `Tax_fed(t)`, `Tax_st(t)` | Taxes | Federal and state (**v5**: columns `tax_federal`, `tax_state`) |
| `τ_marg`, `τ_eff` | Marginal/effective rates | Federal + state |
| `IRMAA_tier(t)`, `IRMAA_sur(t)` | IRMAA tier and surcharge | From lookback year (**v6**: columns `irmaa_tier`, `health_irmaa`) |
| `Gap(t)` | Funding shortfall | Unfunded spending |
| `Lock(t)` | Locked accounts | Owners' accounts still before 59½ (**v1**) |
| `Pen_ewp(t)` | Early-withdrawal penalty paid | 10% of locked-account draws (**v1**) |
| `Surplus(t)` | Surplus income saved | Income beyond spending plus tax, added to cash (**v1**) |

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
| `IRMAA_life` | Lifetime IRMAA | Total surcharges (**v6**: `SimResult.lifetime_irmaa`, printed by `run.py`) |
| `P_succ` | Probability of success | Fraction of paths funded |
| `ε_W` | Solver tolerance | Stopping criterion |

## 5b. Testing approach

Run from the project folder: `python -m unittest` (fast loop) · `SLOW_TESTS=1 python -m unittest -v` (everything; run before trusting a change) · `UPDATE_GOLDEN=1 python -m unittest` (re-record the golden file — only after an *intended* change to the numbers, e.g. a corrected tax table; review the diff first).

Layers of protection, in order of leverage:
1. **Accounting identity matrix:** every strategy × tax on/off × household (unlocked, locked, couple): `gross + SS − tax − penalty − surplus + shortfall = spending` on every row, and no negative draws/balances.
2. **Monotonicity property test:** more spending never reduces shortfall or raises the final balance. The bisection solver is only valid if this holds, so any new strategy must pass it.
3. **Golden regression:** one representative couple (penalties, SS, purchases, curve) compared against a committed JSON; catches "every number shifted a little" drift when adding features.
4. **Closed-form checks:** zero-everything division, annuity formula, SS months, depletion year — with strategy/order pinned explicitly so default changes cannot silently change what is measured.
5. **Mechanism tests:** assert *why* a result differs (e.g. cost basis lowers W *because* capital-gains tax appears), not just that it differs.

Conventions: tests build `Params` through `make_params()` which lists every field explicitly and fails if `Params` gains a field it does not know about. Output rows are rounded to cents, so row-level identities use a ~$0.03 tolerance; the full-precision identity (1e-6) is tested by calling the per-year solver directly. A suite-wide check was done by injecting five deliberate bugs into a scratch copy; all were caught.

## 6. Scope

**v0 (built):** TOML config loader; combined buckets; single nominal return and inflation; federal ordinary + LTCG tax with standard deduction, indexed by inflation; fixed withdrawal order; after-tax gross-up; bisection solver for W; flat and curve spending; dated real-dollar purchases; CSV output; closed-form unit tests.

**v1 (built):**
- Social Security for both spouses, taxed by the provisional-income formula (0/50/85% tiers, unindexed thresholds). (v1: starts at FRA; see v2 for variable claim age.)
- Retirement balances split by person (`traditional_person1/2`, `roth_person1/2`).
- 59½ rule: locked accounts are a last resort; a 10% penalty is computed, included in the after-tax gross-up, and flagged in the console and CSV.
- Withdrawal strategies: `fixed_order`, `taxable_std_then_cash` (plan 2a), `trad_std_then_cash` (comparison plan). `run.py` prints a side-by-side strategy comparison.
- Surplus Social Security income is saved to cash.

**v2 (built):**
- Per-person Social Security claim age (`claim_age_person1/2 = [years, months]`) with early-claim reductions and delayed-retirement credits (see §4); start month and first-year proration follow the claim age.
- Console table adds Spend(nom), marks each person's first benefit year, and labels every column nominal/real. Rows gain a `price_index` column.
- v1 already prorated the first benefit year by month; v2 makes that visible in the table and summary.

**v3 (built):**
- Death dates per person (config, optional; default = 95th birthday). The plan horizon is the last death.
- Joint filing through the year of the first death, single afterwards (standard deduction, brackets, LTCG brackets, Social Security thresholds, and the `"std"` withdrawal cap all switch).
- Survivor Social Security rules (see §4); household benefit schedule computed once per household and cached.
- Deceased's accounts unlocked for the survivor. New row columns: `filing_status`, `n_living`. Console shows deaths, filing, survivor benefit, and marks the first-death and single-filer years.
- Template birth years set to 1972 (so the temporary $6,000 senior deduction, 2025–2028, never applies). `[model] horizon_person` was removed; a non-default value is an error that points to death dates.

**v8 (built):**
- Spousal Social Security benefit (see §4): `rules.spousal_topups`, `rules.ss_real_by_year_detail` (household schedule with the top-up split out), `rules.spousal_claim_factor`; config switch `spousal_benefit`; column `ss_spousal`; `run.py` explains who gets a top-up and when. Verified against an independent calculation across a sweep of claim-age pairs.
- Asset classes: `model.AssetClass`, `Params.asset_classes` / `allocations`, cached per-account rates (`model._bucket_rates`), yearly taxed investment income (`model._investment_income`), reinvested income raising basis, per-account growth. Columns `inv_income_ordinary`, `inv_income_qualified`. `YearTax` gained `inv_ordinary` / `inv_qualified`. `nominal_return` is optional when classes are defined.
- The template now has a spousal example (second benefit $1,200) and fake asset classes; the golden scenario exercises both.

**v7 (built):**
- Two optional inheritance events (father, mother) with a death date each, an inherited traditional IRA under the SECURE Act 10-year rule with the annual-RMD requirement (IRS final regulations: parent past RMD start), and a list of assets (houses, land, belongings, cash) sold some months after death with selling costs and stepped-up basis. Two new buckets (`inh_f`, `inh_m`); six new columns (`inheritance_ira_received`, `inheritance_cash_received`, `inherited_ira_required`, `draw_inherited_ira`, `end_inh_f`, `end_inh_m`); `SimResult.lifetime_inheritance`. `run.py` prints the inheritance summary, notes receipt/sale/deadline years in the table (the `InhIRA(nom)` column replaced `SSTaxed(nom)` in the console; `ss_taxable` remains in the CSV), and reports W without inheritances.
- New pure functions `model._inheritance_inflows` and `model._inherited_forced`; `rules.SINGLE_LIFE` (IRS Table I, 2022+, verified against several sources and the IRS worked example).

**v6 (built):**
- Marketplace premiums before 65, Medicare Part B plus IRMAA (2026 CMS tiers: $81.20–$487.00 Part B and $14.50–$91.00 Part D per person per month) from 65, month-exact, per person, with death handling and optional extra premium growth. IRMAA lookback uses the model's own AGI two years back. Nine new columns (`marketplace_person_months`, `medicare_person_months`, `health_marketplace`, `health_medicare`, `irmaa_lookback_magi`, `irmaa_tier`, `health_irmaa`, `health_total`, `need_nominal`); `SimResult.lifetime_health` and `lifetime_irmaa`. Console table shows `Health(nom)` (the penalty column moved to the notes/CSV), marks Medicare start and IRMAA tier changes.
- `model._health_costs` is a pure function of (params, year, status, scale, lookback MAGI); tests cover it directly.

**v5 (built):**
- State income tax: Indiana (flat 3.05% 2024, 3.00% 2025, 2.95% 2026, 2.9% 2027+; $1,000 personal exemption per person, +$1,000 for 65+; Social Security exempt; IRA withdrawals and capital gains taxed at the flat rate), Texas and Florida (none); table structure ready for more states. Optional `local_tax_rate` for county taxes. Optional move with month proration. New columns: `realized_gains`, `tax_state`, `state_residence`, `state_origin_months`; `marginal_ordinary_rate` is now federal + state; `tax_pct_of_cash_in` includes state tax. `SimResult.lifetime_state_tax`.
- `data_dictionary.txt` written by `run.py` (see §4). Console table adds `StTax(nom)`; the strategy comparison shows federal + state tax.
- `YearTax` gained a `state_tax` field (default 0); `_costs` and `_solve_year` take an optional `state_fn` (from `_state_tax_fn`).

**v4 (built):**
- RMDs from the IRS Uniform Lifetime Table (verified against IRS Pub 590-B, Table III, 2022+ divisors), SECURE 2.0 start ages, per-person accounts, forced first, taxed as ordinary income, surplus saved to cash, standard-deduction cap aware. Survivor rollover and death-year RMD handled. New column `rmd_required`; console table shows `RMD(nom)` and marks each person's first RMD year.
- `model._solve_year` gained an optional `forced` argument (committed draws); `model._rmd_forced` computes them. Verified by an accounting-identity matrix that includes RMD-age households, plus bug-injection testing.

**Known simplifications (current):**
- RMDs: uniform table only (the Joint Life table applies when the sole-beneficiary spouse is more than 10 years younger; it gives smaller RMDs, so ignoring it is conservative and `run.py` prints a note when it would apply). The first RMD is taken in the year the start age is reached (no April-1 delay). No qualified charitable distributions. People born before 1951 are treated as starting at 72 (no 70½ rule). Excess RMD cash is saved to the cash bucket (no tax drag on its growth). Inherited-IRA RMDs (parents) come with the inheritance version.
- Asset classes: constant-mix rebalancing (tax-free, no turnover or capital-gain distributions inside funds); one return per class (no volatility yet: Monte Carlo will add it); tax-exempt income and muni bonds are not modeled; if no classes are defined, one return for all accounts and no yearly tax on income (the older behavior). Individual holdings and tax lots are not modeled.
- **Household spending does not drop after the first death** (a survivor's spending is usually lower); the single filer still spends the full W. A `survivor_spending_fraction` knob is the obvious next addition.
- Spousal benefit: divorced-spouse benefits and earnings-test effects are not modeled; the survivor benefit uses the deceased's record. Survivor benefits use the retirement FRA table (SSA's survivor table differs slightly for some cohorts). No estate/inheritance tax; no RMDs for the inherited balances beyond the inherited-IRA rules.
- Only the temporary $6,000 senior deduction is avoided by the 1972 birth year; the permanent additional standard deduction for age 65+ (about $1,650 per person joint, $2,050 single; verify) is still not modeled.
- No senior deduction, extra standard deduction for age 65+, or NIIT.
- **Inheritance:** the timing of each parent's death is an input, not a forecast (a scenario grid, minimax regret and Monte Carlo timing are the planned ways to handle that uncertainty). Not modeled: inherited Roth IRAs; a spouse (instead of you) inheriting first, so each event is simply what reaches you; estate or inheritance taxes (none federally at these sizes; Indiana has none; another state might); appreciation or income on property between death and sale; keeping and renting a house; probate delays beyond `sale_months_after_death`; what happens if the parent dies before reaching RMD age (rejected as an error). Heir = person 1 only. Receipt at the start of the year is a simplification (assets and IRA arrive earlier on average than they really do).
- **Health costs:** ACA premium tax credits (income-dependent subsidies) are not modeled, so marketplace premiums are a flat price; deductibles, copays, out-of-pocket costs, HSAs, long-term care, Medicare Advantage and the Part B late-enrollment penalty are not modeled; everyone on Medicare is assumed to carry Part D (so pays its IRMAA); IRMAA uses the model's AGI (no tax-exempt interest) and ignores Form SSA-44 appeals; Medicare begins the month of the 65th birthday (SSA's born-on-the-1st rule ignored); the $1,500 marketplace premium is a placeholder. Whether health premiums belong inside or outside W was a choice: they are outside.
- **State tax:** county tax is a single constant rate you supply (Indiana's 2025 law SEA 1 changes how local income taxes are set; details not verified, so it is held constant). Indiana's small credits (age 65+/low income), public-pension and military deductions, and part-year-resident sourcing rules beyond month proration are not modeled. State tax is not deducted federally (standard deduction assumed). Indiana's extra exemption for age 65+ was $1,000 in the source used (another source said $1,500) — verify. Only IN/TX/FL are built in.
- Roth withdrawals before 59½ are penalized on the whole amount (conservative: contributions can in reality come out penalty-free). Roth contribution basis is not tracked.
- No age-based penalty exceptions (Rule of 55, 72(t)/SEPP).
- Social Security: benefit stays constant in real terms. Start-month convention ignores SSA's payment lag (benefits arrive the month after), the born-on-the-1st/2nd rule, and the "62 throughout the month" rule; the earliest claim is 62y0m. Annual income is received at the start of the year like other cash flows.
- **Claim-age results are biased toward delaying until death dates exist:** with a fixed horizon of 95, W rises with claim age (about +$300/mo from 62→67 and +$170/mo from 67→70 in the test scenario) and the advantage shrinks as portfolio returns rise. Re-test with early-death scenarios and Monte Carlo deaths.
- Spending curve keys off person 1's age.
- Year-based ages; ending balance measured at end of the year before the horizon birthday year.
- Traditional money is drawn when the plan says so, or when an RMD forces it (v4).

**Finding from v7 (template data, fake numbers):** inheritances (father 2040, mother 2046; $1.2M of IRAs plus $730k of property and cash, today's dollars) lift W from $7,441 to $10,480/month. **W is set by the tightest year BEFORE the first inheritance arrives**, so an inheritance that arrives after that crunch cannot raise it: with the father at 2040, W stays $10,480 whether the mother dies in 2040, 2046 or 2052, and the earlier of the two events drives the result (father 2032 / mother 2036: $12,169; father 2048 / mother 2052: $8,788). The inherited-IRA policy leaves W unchanged but not taxes: "min" (lump in year 10) paid about $108k more federal tax, $7k more state tax and $39k of IRMAA than "even" and ended $86k lower in real terms. Monotonicity of shortfall in W was stress-tested with large forced distributions (3,000 simulations, no violations).

**Finding from v6 (template data, fake numbers):** health premiums cost the template couple about $1,411/month of W ($8,852 → $7,441): $428k of marketplace premiums before 65 and $277k of standard Medicare after, $705k lifetime (nominal); IRMAA never triggers because AGI stays low. In a high-balance variant ($5.2M traditional, larger Social Security) W falls from $21,501 to $19,892 and IRMAA surcharges alone total $154k over 30 of 40 years (tier 3 at first Medicare year, tiers 1–2 afterwards), driven by RMDs and Social Security: the income-driven cost that Roth conversions could target. Monotonicity of shortfall in W was stress-tested with IRMAA in play (4,800 simulations, no violations).

**Finding from v5 (template data, fake numbers):** W per month: no state tax $9,010; Indiana state tax only $8,852 (−$158); Indiana plus a 1.7% county tax $8,759 (−$251); plus a move to a no-tax state in mid-2032 $8,983 (−$27); a move in mid-2045 $8,838 (−$171). Lifetime state tax $111k (state only), $175k (with county), $7.8k (county, moving in 2032). The county tax raises the Indiana burden by more than half again; moving earlier matters far more than moving later.

**Finding from v4 (template data):** in the shipped template (couple retiring at 54 with only $350k outside retirement accounts) all three withdrawal strategies give identical results, because the non-retirement money is gone before 59½ and every plan then draws the same retirement money. That is an artifact of the template, not a bug. In a variant with $1M taxable (non-retirement money lasting past 59½) and the same retirement balances, `trad_std_then_cash` (fill the standard deduction with traditional money early) cut lifetime federal tax by about 11% ($1.01M vs $1.13M) and ended about 2% higher at the same spending, while W moved only +0.16%: W is a blunt metric for tax strategy, so keep lifetime tax and ending balance alongside it. Fake numbers; re-test with real data.

**Finding from v1 (template data):** the "up to one standard deduction from taxable" plan (2a) produced the same W as plain cash→taxable→traditional→Roth. Reason: taxable-account withdrawals are mostly basis, and gains already enjoy a 0% long-term-gains bracket up to roughly $131k of total income (MFJ), so the standard deduction is not the scarce resource for taxable-account money. It is scarce for *ordinary* income, which is why the comparison plan fills it with traditional withdrawals. Re-test once RMDs, inheritance, and Roth conversions exist, because the ranking can change.

## 7. Roadmap (planned build order)

1. ~~Social Security + its taxation (provisional income)~~ — **done in v1**; variable claim age **v2**; survivor **v3**; spousal benefit **v8**. Still to do: a claim-age sweep/comparison (now meaningful: death dates, survivor and spousal rules exist)
2. ~~First-death / single-filer switch~~ — **done in v3**; ~~RMDs~~ — **done in v4**. Still to do: Joint Life table, QCDs, inherited-IRA RMDs (with item 4)
3. Roth conversions (bracket-filling, then IRMAA ceiling with 2-year lookback — the lookback and tier machinery now exist as of v6) — **next**. (The temporary senior deduction is moot for the 1972 cohort; the extra 65+ standard deduction is still to add.)
4. ~~Inheritance events + inherited IRA rules~~ — **done in v7** (father and mother, with 10-year rule and asset sales). Still to do: the scenario grid / minimax regret / Monte Carlo timing, inherited Roth, Joint Life and QCD refinements
5. ~~Healthcare premiums~~ — **done in v6** (marketplace, Medicare, IRMAA). Still to do: long-term care, out-of-pocket costs, ACA subsidies, spending curve refinements
6. ~~State taxes and the move date~~ — **done in v5** (Indiana, Texas, Florida; more states = new table entries)
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
| 2026-10-22 | Retirement balances split by person so the 59½ rule uses each owner's age (also needed later for RMDs, survivor, inheritance). |
| 2026-10-22 | SS benefit entered as monthly amount at FRA in start-year dollars; starts at FRA; COLA = inflation; surplus income saved to cash. |
| 2026-10-22 | Penalty-free access starts the calendar year of the 59½ date if on/before June 30, else the next year (annual-model convention). |
| 2026-10-22 | Locked retirement money is a last resort; early_access_order default Roth first (penalty only in model) then traditional. |
| 2026-10-22 | Withdrawal plans are step lists `(category, cap)`; built-ins: fixed_order, taxable_std_then_cash, trad_std_then_cash. |
| 2026-10-24 | `model.py` is now the user-commented master copy. Future edits are applied as small patches to it (never regenerated), so the user's comments survive. Executable code verified identical (AST comparison) when adopted. |
| 2026-10-24 | v2: Social Security claim age per person as `[years, months]` in config (default FRA), window 62y0m–70y0m; early-claim reduction 5/9% (first 36 mo) then 5/12%; delayed credit 2/3% per month to 70. |
| 2026-11-03 | v8: spousal benefit = (50% x other's PIA - own PIA) x early-start reduction, starting when BOTH have claimed, based on the PIA (never the delayed check), ends at first death; switch `spousal_benefit`. Works for any pair of claim ages 62-70. |
| 2026-11-03 | v8: asset classes (return, income yield, qualified share) with one mix per account; constant-mix rebalancing; dividends/interest taxed yearly in cash and taxable, reinvested, added to basis; optional (no classes = old single-rate behavior). |
| 2026-11-03 | Verification: 22 deliberate-bug checks on spousal and asset classes all caught (one only by the golden file, now also by a targeted test). Heavy bug-injection runs are NOT part of routine work; run the test suite locally instead. |
| 2026-10-29 | v7: inheritances modeled as an inherited traditional IRA (SECURE Act 10-year rule with annual RMDs, because the parents were past their RMD start) plus a list of assets sold after death; the parents' death dates are config inputs. |
| 2026-10-29 | IRA arrives at the start of the year of death; the parent's own RMD for that year is due; years 1-9 use the Single Life Table at the heir's age in year 1 minus one per year; year 10 empties the account. Policy "even" (default) vs "min". |
| 2026-10-29 | Inherited property: stepped-up basis, so net sale proceeds (after selling costs) go to the taxable bucket at full basis; amounts are today's dollars and keep pace with inflation. A parent must have been at RMD age at death (otherwise an error). |
| 2026-10-29 | Inherited IRAs belong to the "trad" category and are drawn before own traditional accounts; their forced draws count against the "std" cap; no 10% penalty. |
| 2026-10-28 | v6: health premiums are an extra required expense on top of W (W excludes them). Marketplace per person before 65; Medicare Part B + IRMAA from the month of the 65th birthday. |
| 2026-10-28 | IRMAA: Parts B and D, two-year lookback on modeled AGI, thresholds of the CURRENT filing status (widow's trap modeled, SSA-44 appeal not), top threshold fixed, everyone on Medicare pays it. 2026 CMS tables checked against several sources. |
| 2026-10-28 | Health inputs are per person in start-year dollars (default $1,500 marketplace is a user-confirmed placeholder); premium growth above inflation defaults to 0. No `[healthcare]` block = no health costs. |
| 2026-10-27 | v5: state income tax for Indiana (flat, by-year schedule, exemptions, Social Security exempt) plus TX/FL (none); optional move date/state with month proration; optional `local_tax_rate`; no `[state]` block = no state tax. |
| 2026-10-27 | State tax enters the solver's after-tax cost, so W is after federal AND state tax. |
| 2026-10-27 | County tax defaults to 0 in the template (not assumed for the user); the rate varies by county and must be supplied. |
| 2026-10-27 | Every run writes `data_dictionary.txt`; documentation lives in `columns.py` and is enforced by tests. |
| 2026-10-26 | v4: RMDs implemented from the IRS Uniform Lifetime Table; start age 73 (born 1951–59) / 75 (born 1960+); per-person, own accounts only; taken first, ordinary income, excess saved to cash. |
| 2026-10-26 | RMD rules checked against IRS Pub 590-B (table values, examples) before coding; Joint Life table deliberately deferred (conservative; irrelevant for same-age couples). |
| 2026-10-26 | Survivor assumed to roll the deceased's account into their own; deceased's death-year RMD still taken. |
| 2026-10-25 | v3: death dates per person in config (default 95th birthday); horizon = last death; `horizon_person` removed. |
| 2026-10-25 | Filing: joint through the year of the first death (IRS rule), single after. Social Security: survivor gets the larger check from the death month, with basis/widow-floor/age-60/early-reduction rules. |
| 2026-10-25 | Deceased's accounts are the survivor's and are penalty-free from the year after death (spousal beneficiary treatment). |
| 2026-10-25 | Survivor spending drop deliberately NOT modeled yet (kept as an explicit known simplification). |
| 2026-10-25 | Template birth years set to 1972; temporary $6,000 senior deduction ends 2028 so it is irrelevant for this cohort. |
| 2026-10-24 | Console tables show Spend(real) and Spend(nom) side by side; all dollar columns labeled nominal/real; `price_index` column added for deflating any series. |
| 2026-10-24 | Plotting is planned but not built; data stays plot-ready (see §10). |
| 2026-10-24 | Test suite restructured (explicit `make_params`, pinned defaults, accounting-identity matrix, monotonicity property test, golden regression, edge cases, slow-test marker). |
| 2026-10-24 | Finding: with equal returns, withdrawal order barely matters without Social Security; with SS, trad-first beats Roth-first (~$96/mo in test scenario) because it avoids stacking on taxable benefits. |
| 2026-10-22 | Plan 2a is kept but flagged: no W advantage over fixed order in the template run (see Finding in §6). |

## 9. Open Questions

- Parents' and own ages (kept private; supply via config only) to set realistic inheritance-timing scenarios.
- Whether legacy for children matters (decides whether to add a minimum-bequest constraint early).
- Verify 2026 tax tables (standard deduction sources conflicted: $30,000 vs $32,200 for MFJ; the model uses $32,200).
- Roth contribution basis (penalty-free withdrawals before 59½): worth tracking if either spouse may retire early.
- Add the permanent extra standard deduction for age 65+ (the temporary senior deduction is moot for the 1972 cohort).
- Inheritance: a scenario grid over the two death dates (and minimax regret) is the natural next tool; inherited Roth; whether a surviving parent inherits first; estate or inheritance tax in the parents' state; sensible default for the `share` if siblings inherit.
- Health: ACA premium tax credits (a big lever for early retirees, and tied to MAGI); Medigap and Part D premium defaults; out-of-pocket costs and long-term care; Medicare Part B premium growth vs. inflation; verify 2026 CMS figures and each fall's new ones.
- Verify Indiana's extra age-65 exemption ($1,000 vs $1,500) and your county's income tax rate; decide whether to model Indiana's age-65/low-income credit.
- More states: add entries to `statetax.STATES` (flat-rate states are one line each; progressive states need a bracket function).
- RMD refinements to consider: Joint Life table when a spouse is >10 years younger; QCDs; reinvesting surplus RMD cash into the taxable account (with basis) instead of the cash bucket.
- Should survivor spending fall after the first death (e.g. to 70–80%)? Which fraction, and does the single filer's spending curve change?
- Consider returning unrounded rows from `simulate()` (round only when writing CSV) so row-level tests can use 1e-6 instead of ~$0.03. Needs a small `model.py` edit — do it on the user's commented copy.
- Should the strategy layer target tax brackets (e.g., fill the 12% bracket) rather than the standard deduction? Natural next step with Roth conversions.

## 10. Plotting plan (not built — design constraints to keep)

Plots are a presentation layer on top of the engine, never inside it.

**Data contract (what plots will read):**
- `simulate()` rows / the CSVs in `outputs/` are the plot interface: tidy, one row per year, consistent column names. Keep them stable; add columns rather than renaming (the golden test will flag any change).
- Nominal unless the column ends in `_real`; `price_index` deflates any nominal column; per-bucket balances and draws are separate columns so stacked charts need no recomputation.
- Anything a plot needs should be a column or a pure function of the columns. No plot should re-implement tax or withdrawal logic.

**Code layout:** a future `plots.py` reads the CSV (or `SimResult.rows`); `run.py` stays text-only. Core model stays standard-library only; matplotlib becomes an *optional* dependency imported inside `plots.py`.

**Candidate plots, roughly in priority order:**
1. Income sources vs. spending, by year (stacked bars of Social Security + draws by bucket, line for Spend(nom)) — shows when benefits do/do not cover spending.
2. Account balances by bucket over time (stacked area, real dollars).
3. Taxes by year with marginal rate and Social Security taxable share (shows the tax "torpedo" zone).
4. W vs. claim age (one line per strategy); W by strategy bars.
5. Later: Monte Carlo fan charts (10/50/90th percentile balances), probability of success vs. spending, inheritance-timing heat map of W/regret.
