"""
model.py -- Retirement / tax model, v0.

Layers (see DESIGN.md):
  1. Inputs     -> Params + load_params()          (reads your local TOML file)
  2. Engine     -> simulate()                      (year-by-year, deterministic)
  3. Strategy   -> withdrawal_order in Params      (more strategies come later)
  4. Solver     -> solve_max_w()                   (finds W by bisection)

Conventions
  * Annual steps. Withdrawals happen at the START of each year, then balances
    grow for the year.
  * "Real" dollars = dollars of Params.start_year.
  * W is monthly, REAL, AFTER-TAX household spending.
  * Year k = start_year + k. Ages are year-based (year - birth year).
  * The model ends when the horizon person reaches horizon_age: the final
    simulated year is the one before that birthday year, and the ending balance
    is measured at the end of that year.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from typing import Optional

import taxes

BUCKETS = ("cash", "taxable", "trad", "roth")
SUCCESS_TOLERANCE = 0.01  # dollars of unfunded spending we ignore


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
    tax_enabled: bool
    withdrawal_order: tuple        # subset/permutation of BUCKETS
    cash: float
    taxable_value: float
    taxable_basis: float
    traditional: float
    roth: float
    target_monthly_real: float
    curve: tuple                   # ((age, multiplier), ...) person1's age
    purchases: tuple               # ((year, amount_real, label), ...)

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
        return self.cash + self.taxable_value + self.traditional + self.roth


def load_params(path: str) -> Params:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    hh, m, b, s = c["household"], c["model"], c["balances"], c["spending"]

    order = tuple(m.get("withdrawal_order", ["cash", "taxable", "trad", "roth"]))
    bad = [k for k in order if k not in BUCKETS]
    if bad:
        raise ValueError(f"Unknown bucket(s) in withdrawal_order: {bad}")
    if hh["filing_status"] not in taxes.STANDARD_DEDUCTION:
        raise ValueError("filing_status must be 'mfj' or 'single'")

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
        withdrawal_order=order,
        cash=float(b.get("cash", 0)),
        taxable_value=float(b.get("taxable_value", 0)),
        taxable_basis=float(b.get("taxable_basis", 0)),
        traditional=float(b.get("traditional", 0)),
        roth=float(b.get("roth", 0)),
        target_monthly_real=float(s.get("target_monthly_real", 0)),
        curve=tuple((float(a), float(x)) for a, x in s.get("curve", [[0, 1.0]])),
        purchases=tuple((int(x["year"]), float(x["amount_real"]), x.get("label", ""))
                        for x in c.get("purchases", [])),
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
    depletion_year: Optional[int]

    @property
    def success(self) -> bool:
        return self.shortfall_total <= SUCCESS_TOLERANCE


def _draw_plan(x: float, state: dict, order: tuple):
    """Spread a total gross draw x across buckets in the given order."""
    remaining, draws = x, {k: 0.0 for k in BUCKETS}
    for key in order:
        d = min(remaining, state[key])
        draws[key] = d
        remaining -= d
    return draws


def _tax_for(draws: dict, state: dict, basis: float, scale: float, p: Params):
    if not p.tax_enabled:
        return 0.0, draws["trad"], 0.0
    gain_frac = max(0.0, 1.0 - basis / state["taxable"]) if state["taxable"] > 0 else 0.0
    gains = draws["taxable"] * gain_frac
    tax = taxes.federal_tax(draws["trad"], gains, p.filing_status, scale)
    return tax, draws["trad"], gains


def _solve_year(need: float, state: dict, basis: float, scale: float, p: Params):
    """Find the smallest gross draw whose after-tax proceeds cover `need`.
    Net proceeds rise monotonically with the gross draw (marginal tax < 100%),
    so bisection is safe."""
    available = sum(state[k] for k in p.withdrawal_order)

    def net(x):
        draws = _draw_plan(x, state, p.withdrawal_order)
        tax, ordinary, gains = _tax_for(draws, state, basis, scale, p)
        return sum(draws.values()) - tax, draws, tax, ordinary, gains

    n_all = net(available)
    if n_all[0] <= need:
        return n_all[1], n_all[2], n_all[3], n_all[4], max(0.0, need - n_all[0])
    lo, hi = 0.0, available
    for _ in range(200):
        mid = (lo + hi) / 2
        if net(mid)[0] >= need:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    n = net(hi)
    return n[1], n[2], n[3], n[4], 0.0


def simulate(p: Params, w_monthly_real: float, use_curve: bool = False,
             include_purchases: bool = True) -> SimResult:
    state = {"cash": p.cash, "taxable": p.taxable_value,
             "trad": p.traditional, "roth": p.roth}
    basis = p.taxable_basis
    rows, shortfall_total, lifetime_tax, depletion = [], 0.0, 0.0, None

    for k in range(p.n_years):
        year = p.start_year + k
        price = (1 + p.inflation) ** k                    # to nominal from start_year $
        scale = (1 + p.inflation) ** (year - taxes.BASE_YEAR)  # bracket indexing
        age1 = year - p.person1_birth.year
        mult = curve_multiplier(p.curve, age1) if use_curve else 1.0
        purchases_real = sum(a for (y, a, _) in p.purchases if y == year) if include_purchases else 0.0
        spend_real = 12.0 * w_monthly_real * mult + purchases_real
        need = spend_real * price

        draws, tax, ordinary, gains, short = _solve_year(need, state, basis, scale, p)

        if state["taxable"] > 0 and draws["taxable"] > 0:
            basis *= 1.0 - draws["taxable"] / state["taxable"]
        for key in BUCKETS:
            state[key] -= draws[key]
            state[key] *= 1.0 + p.nominal_return          # growth after withdrawal
        shortfall_total += short
        lifetime_tax += tax
        if short > SUCCESS_TOLERANCE and depletion is None:
            depletion = year

        gross = sum(draws.values())
        end_total = sum(state.values())
        rows.append({
            "year": year,
            "age_person1": age1,
            "spend_real": round(spend_real, 2),
            "spend_nominal": round(need, 2),
            "gross_withdrawal": round(gross, 2),
            "tax_federal": round(tax, 2),
            "eff_tax_rate_on_withdrawal": round(tax / gross, 4) if gross > 0 else 0.0,
            "marginal_ordinary_rate": round(taxes.marginal_rate(
                ordinary, gains, p.filing_status, scale), 4) if p.tax_enabled else 0.0,
            "shortfall_nominal": round(short, 2),
            "end_cash": round(state["cash"], 2),
            "end_taxable": round(state["taxable"], 2),
            "end_trad": round(state["trad"], 2),
            "end_roth": round(state["roth"], 2),
            "end_total_nominal": round(end_total, 2),
            "end_total_real": round(end_total / (price * (1 + p.inflation)), 2),
        })

    final_nominal = sum(state.values())
    final_real = final_nominal / (1 + p.inflation) ** p.n_years
    return SimResult(rows, final_nominal, final_real, shortfall_total,
                     lifetime_tax, depletion)


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
