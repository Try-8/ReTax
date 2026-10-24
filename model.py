"""
model.py -- Retirement / tax model, v1.

Layers (see DESIGN.md):
  1. Inputs     -> Params + load_params()          (reads your local TOML file)
  2. Engine     -> simulate()                      (year-by-year, deterministic)
  3. Strategy   -> STRATEGIES (withdrawal plans)   (more strategies come later)
  4. Solver     -> solve_max_w()                   (finds W by bisection)

Conventions
  * Annual steps. Withdrawals and Social Security arrive at the START of each
    year, then balances grow for the year.
  * "Real" dollars = dollars of Params.start_year.
  * W is monthly, REAL, AFTER-TAX household spending.
  * Year k = start_year + k. Ages are year-based (year - birth year).
  * The model ends when the horizon person reaches horizon_age: the final
    simulated year is the one before that birthday year, and the ending balance
    is measured at the end of that year.

Withdrawal plans (v1)
  A plan is an ordered list of steps (category, cap). Each year the engine
  draws from step 1 up to its cap, then step 2, and so on, until the household's
  AFTER-TAX spending need is met. Categories: cash, taxable, trad, roth.
  cap = None (no cap) or "std" (the inflation-indexed standard deduction).
  Retirement accounts of a person younger than 59.5 are LOCKED: they are only
  touched as a last resort (after all strategy steps), and a 10% penalty applies.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from math import inf
from typing import Optional

import rules
import taxes

BUCKETS = ("cash", "taxable", "trad1", "trad2", "roth1", "roth2")
CATEGORIES = ("cash", "taxable", "trad", "roth")
PENALTY_RATE = 0.10          # early-withdrawal penalty
SUCCESS_TOLERANCE = 0.01     # dollars of unfunded spending we ignore

# name -> steps. "fixed_order" is built from Params.fixed_order instead.
STRATEGIES = {
    "fixed_order": None,
    # Plan 2a: up to one standard deduction from the taxable account, the rest
    # from cash; once cash is gone: remaining taxable, then traditional, then Roth.
    "taxable_std_then_cash": (("taxable", "std"), ("cash", None), ("taxable", None),
                              ("trad", None), ("roth", None)),
    # Comparison plan: fill the standard deduction with traditional-account
    # money, then cash, taxable, more traditional, Roth last.
    "trad_std_then_cash": (("trad", "std"), ("cash", None), ("taxable", None),
                           ("trad", None), ("roth", None)),
}


# --------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    start_year: int
    person1_birth: date
    person2_birth: Optional[date]
    horizon_age: int
    horizon_person: str            # "person1" | "person2" | "younger"
    filing_status: str             # "mfj" | "single"
    nominal_return: float
    inflation: float
    tax_enabled: bool              # False disables income tax AND penalties (testing)
    cash: float
    taxable_value: float
    taxable_basis: float
    trad1: float                   # traditional IRA/401k, person 1
    trad2: float
    roth1: float                   # Roth, person 1
    roth2: float
    target_monthly_real: float
    curve: tuple                   # ((age, multiplier), ...) person1's age
    purchases: tuple               # ((year, amount_real, label), ...)
    strategy: str = "fixed_order"
    fixed_order: tuple = ("cash", "taxable", "trad", "roth")
    early_access_order: tuple = ("roth", "trad")   # last-resort order for locked money
    ss_fra_monthly1: float = 0.0   # monthly benefit at FRA, start-year dollars
    ss_fra_monthly2: float = 0.0
    ss_multiplier: float = 1.0     # sensitivity knob, e.g. 0.77 for a benefit cut

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Choose from {list(STRATEGIES)}")
        bad = [c for c in self.fixed_order if c not in CATEGORIES]
        if bad:
            raise ValueError(f"Unknown category in fixed_order: {bad} (use {CATEGORIES})")
        bad = [c for c in self.early_access_order if c not in ("trad", "roth")]
        if bad:
            raise ValueError(f"early_access_order may only contain 'trad' and 'roth': {bad}")
        if self.person2_birth is None and (self.trad2 or self.roth2 or self.ss_fra_monthly2):
            raise ValueError("Person 2 has balances/benefits but no person2_birth was given.")

    # ---- derived ----
    @property
    def horizon_birth_year(self) -> int:
        b1 = self.person1_birth.year
        b2 = self.person2_birth.year if self.person2_birth else b1
        if self.horizon_person == "person1":
            return b1
        if self.horizon_person == "person2":
            return b2
        return max(b1, b2)  # "younger"

    @property
    def n_years(self) -> int:
        return self.horizon_birth_year + self.horizon_age - self.start_year

    @property
    def total_start_balance(self) -> float:
        return (self.cash + self.taxable_value + self.trad1 + self.trad2
                + self.roth1 + self.roth2)

    @property
    def _older_first(self) -> tuple:
        if self.person2_birth and self.person2_birth < self.person1_birth:
            return (2, 1)
        return (1, 2)

    @property
    def trad_order(self) -> tuple:
        """Traditional buckets in draw order: older person's account first."""
        return tuple(f"trad{i}" for i in self._older_first)

    @property
    def roth_order(self) -> tuple:
        return tuple(f"roth{i}" for i in self._older_first)

    @property
    def strategy_steps(self) -> tuple:
        if self.strategy == "fixed_order":
            return tuple((c, None) for c in self.fixed_order)
        return STRATEGIES[self.strategy]

    def unlock_year(self, person: int) -> int:
        birth = self.person1_birth if person == 1 else self.person2_birth
        return rules.early_unlock_year(birth) if birth else -10**9

    def locked_buckets(self, year: int) -> frozenset:
        out = set()
        for person in (1, 2):
            if self.unlock_year(person) > year:
                out.update({f"trad{person}", f"roth{person}"})
        return frozenset(out)


def load_params(path: str) -> Params:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    hh, m, b, s = c["household"], c["model"], c["balances"], c["spending"]
    ss = c.get("social_security", {})
    st = c.get("strategy", {})

    if "traditional" in b or "roth" in b:
        raise ValueError(
            "v1 needs retirement balances split by person (the 59.5 rule depends on "
            "each owner's age). Replace 'traditional'/'roth' in [balances] with "
            "traditional_person1/2 and roth_person1/2 -- see config.template.toml.")
    if hh["filing_status"] not in taxes.STANDARD_DEDUCTION:
        raise ValueError("filing_status must be 'mfj' or 'single'")

    default_order = m.get("withdrawal_order", ["cash", "taxable", "trad", "roth"])  # v0 key
    p = Params(
        start_year=int(m["start_year"]),
        person1_birth=hh["person1_birth"],
        person2_birth=hh.get("person2_birth"),
        horizon_age=int(m.get("horizon_age", 95)),
        horizon_person=m.get("horizon_person", "younger"),
        filing_status=hh["filing_status"],
        nominal_return=float(m["nominal_return"]),
        inflation=float(m["inflation"]),
        tax_enabled=bool(m.get("tax_enabled", True)),
        cash=float(b.get("cash", 0)),
        taxable_value=float(b.get("taxable_value", 0)),
        taxable_basis=float(b.get("taxable_basis", 0)),
        trad1=float(b.get("traditional_person1", 0)),
        trad2=float(b.get("traditional_person2", 0)),
        roth1=float(b.get("roth_person1", 0)),
        roth2=float(b.get("roth_person2", 0)),
        target_monthly_real=float(s.get("target_monthly_real", 0)),
        curve=tuple((float(a), float(x)) for a, x in s.get("curve", [[0, 1.0]])),
        purchases=tuple((int(x["year"]), float(x["amount_real"]), x.get("label", ""))
                        for x in c.get("purchases", [])),
        strategy=st.get("name", "fixed_order"),
        fixed_order=tuple(st.get("fixed_order", default_order)),
        early_access_order=tuple(st.get("early_access_order", ["roth", "trad"])),
        ss_fra_monthly1=float(ss.get("fra_monthly_person1", 0)),
        ss_fra_monthly2=float(ss.get("fra_monthly_person2", 0)),
        ss_multiplier=float(ss.get("benefit_multiplier", 1.0)),
    )
    if p.n_years <= 0:
        raise ValueError("Horizon is not after start_year; check birth dates/start_year.")
    return p


def curve_multiplier(curve, age: float) -> float:
    """Piecewise-linear interpolation, flat outside the given ages."""
    pts = sorted(curve)
    if age <= pts[0][0]:
        return pts[0][1]
    if age >= pts[-1][0]:
        return pts[-1][1]
    for (a0, m0), (a1, m1) in zip(pts, pts[1:]):
        if a0 <= age <= a1:
            return m0 + (m1 - m0) * (age - a0) / (a1 - a0)
    return 1.0


# --------------------------------------------------------------------------
# 2. Engine
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    rows: list
    final_nominal: float
    final_real: float
    shortfall_total: float
    lifetime_tax: float
    lifetime_penalty: float
    penalty_years: list
    depletion_year: Optional[int]

    @property
    def success(self) -> bool:
        return self.shortfall_total <= SUCCESS_TOLERANCE


def _expand(category, locked, p, want_locked):
    """Concrete buckets for a category, keeping only locked or only unlocked ones."""
    if category in ("cash", "taxable"):
        return [] if want_locked else [category]
    pool = p.trad_order if category == "trad" else p.roth_order
    return [b for b in pool if (b in locked) == want_locked]


# locked = the bucket names that are still locked this year
def _year_steps(p, locked, scale):
    """This year's list of (buckets, cap_nominal). Locked money comes last."""
    # determine the nominal standard deduction
    std = taxes.STANDARD_DEDUCTION[p.filing_status] * scale
    steps = []

    # Take money out in the order dictated by the strategy
    # p.strategy_steps gives the ordered (category,cap) from the strategy (cash,none),(taxable,none),(trad,std)
    for category, cap in p.strategy_steps:
        # _expand turns a category into concrete buckets, keeping only unlocked ones
        buckets = _expand(category, locked, p, want_locked=False)
        if buckets:
            steps.append((buckets, std if cap == "std" else inf))

    # If there is still a need, take money out of locked buckets (incurring a penalty)
    # If you must take locked money, take Roth first (no penalty on contributions), then traditional
    for category in p.early_access_order:
        buckets = _expand(category, locked, p, want_locked=True)
        if buckets:
            steps.append((buckets, inf))
    return steps

# takes money from buckets respecting caps (from _year_steps) and available balances
def _draw_plan(x, state, steps):
    """Spread a total gross withdrawal requested (pre-tax dollars) x across the plan's steps (monotone in x).
    x - requested withdrawal 
    state - the current balances dict
    steps - the ordered buckets with their caps """
    # remaining = how much of the request is still unfilled, draws starts all zeros
    remaining, draws = x, {b: 0.0 for b in BUCKETS}
    for buckets, cap in steps:
        # use step_left for the cap in this bucket
        step_left = cap
        for b in buckets:
            # take the minimum of the remaining request, the cap amount, or the bucket's unallocated balance
            d = min(remaining, step_left, state[b] - draws[b])
            draws[b] += d
            remaining -= d
            step_left -= d
            if remaining <= 0:
                return draws
    return draws


def _costs(draws, state, basis, ss_gross, scale, p, locked):
    """Income tax and early-withdrawal penalty implied by a set of draws."""
    trad_inc = draws["trad1"] + draws["trad2"]
    if not p.tax_enabled:
        return taxes.YearTax(0.0, 0.0, 0.0, 0.0), 0.0, trad_inc, 0.0
    gain_frac = max(0.0, 1.0 - basis / state["taxable"]) if state["taxable"] > 0 else 0.0
    gains = draws["taxable"] * gain_frac
    yt = taxes.year_tax(trad_inc, gains, ss_gross, p.filing_status, scale)
    penalty = PENALTY_RATE * sum(draws[b] for b in locked)
    return yt, penalty, trad_inc, gains


def _solve_year(need, state, basis, ss_gross, scale, p, steps, locked):
    """Smallest gross draw whose after-tax proceeds (plus Social Security) cover
    `need`. Net proceeds rise monotonically with the draw (marginal tax plus
    penalty stays below 100%), so bisection is safe.
    Returns (draws, YearTax, penalty, trad_inc, gains, surplus, shortfall)."""
    capacity = sum(_draw_plan(inf, state, steps).values())

    def net(x):
        draws = _draw_plan(x, state, steps)
        yt, pen, trad_inc, gains = _costs(draws, state, basis, ss_gross, scale, p, locked)
        return sum(draws.values()) + ss_gross - yt.income_tax - pen, draws, yt, pen, trad_inc, gains

    top = net(capacity)
    if top[0] <= need:                       # cannot fund this year's spending
        return (*top[1:], 0.0, max(0.0, need - top[0]))
    base = net(0.0)
    if base[0] >= need:                      # Social Security alone covers it
        return (*base[1:], base[0] - need, 0.0)
    lo, hi = 0.0, capacity
    for _ in range(200):
        mid = (lo + hi) / 2
        if net(mid)[0] >= need:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    return (*net(hi)[1:], 0.0, 0.0)


def _ss_income(p, year, price):
    """Total Social Security received in `year` (nominal dollars)."""
    total = 0.0
    for birth, monthly in ((p.person1_birth, p.ss_fra_monthly1),
                           (p.person2_birth, p.ss_fra_monthly2)):
        if birth is not None and monthly > 0:
            total += monthly * rules.ss_months_paid(birth, year) * price * p.ss_multiplier
    return total


def simulate(p: Params, w_monthly_real: float, use_curve: bool = False,
             include_purchases: bool = True) -> SimResult:
    state = {"cash": p.cash, "taxable": p.taxable_value, "trad1": p.trad1,
             "trad2": p.trad2, "roth1": p.roth1, "roth2": p.roth2}
    basis = p.taxable_basis
    rows, penalty_years = [], []
    shortfall_total = lifetime_tax = lifetime_penalty = 0.0
    depletion = None

    for k in range(p.n_years):
        # the year being simulated
        year = p.start_year + k
        
        # cumulative inflation factor:  start-year dollars -> nominal dollars
        price = (1 + p.inflation) ** k                          # nominal / start-year $

        # cumulative scale factor for tax brackets, BASE_YEAR = 2026 right now
        scale = (1 + p.inflation) ** (year - taxes.BASE_YEAR)   # bracket indexing
        age1 = year - p.person1_birth.year
        mult = curve_multiplier(p.curve, age1) if use_curve else 1.0

        purchases_real = (sum(a for (y, a, _) in p.purchases if y == year)
                          if include_purchases else 0.0)

        #Annual spending amount, with input of desired montly spending, mult is for curved spending if used
        spend_real = 12.0 * w_monthly_real * mult + purchases_real
        
        #The real dollars converted to nominal dollars
        need = spend_real * price

        # determine which accounts we should remain locked or unlcok them (IRA has penalty before 59.5)
        locked = p.locked_buckets(year)


        steps = _year_steps(p, locked, scale)
        ss_gross = _ss_income(p, year, price)
        draws, yt, pen, trad_inc, gains, surplus, short = _solve_year(
            need, state, basis, ss_gross, scale, p, steps, locked)

        marginal = 0.0
        if p.tax_enabled:
            bumped = taxes.year_tax(trad_inc + 1000.0, gains, ss_gross, p.filing_status, scale)
            marginal = (bumped.income_tax - yt.income_tax) / 1000.0

        if state["taxable"] > 0 and draws["taxable"] > 0:
            basis *= 1.0 - draws["taxable"] / state["taxable"]
        for b in BUCKETS:
            state[b] -= draws[b]
        state["cash"] += surplus                                # unspent income is saved
        for b in BUCKETS:
            state[b] *= 1.0 + p.nominal_return                  # growth after withdrawal

        shortfall_total += short
        lifetime_tax += yt.income_tax
        lifetime_penalty += pen
        if pen > 0.005:
            penalty_years.append(year)
        if short > SUCCESS_TOLERANCE and depletion is None:
            depletion = year

        gross = sum(draws.values())
        end_total = sum(state.values())
        cash_in = gross + ss_gross
        rows.append({
            "year": year,
            "age_person1": age1,
            "spend_real": round(spend_real, 2),
            "spend_nominal": round(need, 2),
            "ss_gross": round(ss_gross, 2),
            "ss_taxable": round(yt.taxable_ss, 2),
            "provisional_income": round(yt.provisional_income, 2),
            "agi": round(yt.agi, 2),
            "draw_cash": round(draws["cash"], 2),
            "draw_taxable": round(draws["taxable"], 2),
            "draw_trad": round(draws["trad1"] + draws["trad2"], 2),
            "draw_roth": round(draws["roth1"] + draws["roth2"], 2),
            "gross_withdrawal": round(gross, 2),
            "tax_federal": round(yt.income_tax, 2),
            "penalty_early_withdrawal": round(pen, 2),
            "early_withdrawal_flag": "PENALTY" if pen > 0.005 else "",
            "tax_pct_of_cash_in": round((yt.income_tax + pen) / cash_in, 4) if cash_in > 0 else 0.0,
            "marginal_ordinary_rate": round(marginal, 4),
            "surplus_to_cash": round(surplus, 2),
            "shortfall_nominal": round(short, 2),
            "end_cash": round(state["cash"], 2),
            "end_taxable": round(state["taxable"], 2),
            "end_trad1": round(state["trad1"], 2),
            "end_trad2": round(state["trad2"], 2),
            "end_roth1": round(state["roth1"], 2),
            "end_roth2": round(state["roth2"], 2),
            "end_total_nominal": round(end_total, 2),
            "end_total_real": round(end_total / (price * (1 + p.inflation)), 2),
        })

    final_nominal = sum(state.values())
    final_real = final_nominal / (1 + p.inflation) ** p.n_years
    return SimResult(rows, final_nominal, final_real, shortfall_total, lifetime_tax,
                     lifetime_penalty, penalty_years, depletion)


# --------------------------------------------------------------------------
# 4. Solver -- metric 1: W
# --------------------------------------------------------------------------
def solve_max_w(p: Params, use_curve: bool = False, include_purchases: bool = True,
                tol: float = 0.01) -> float:
    """Largest monthly real after-tax W such that the money lasts to the horizon.
    With use_curve=True, W is the scale on the spending curve (multiplier 1.0)."""
    if not simulate(p, 0.0, use_curve, include_purchases).success:
        return 0.0  # even W=0 fails (purchases alone exceed the portfolio)
    lo, hi = 0.0, max(p.total_start_balance / 12.0, 1.0)
    for _ in range(80):                       # make sure hi is infeasible
        if not simulate(p, hi, use_curve, include_purchases).success:
            break
        hi *= 2.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if simulate(p, mid, use_curve, include_purchases).success:
            lo = mid
        else:
            hi = mid
    return lo
