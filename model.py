"""
model.py -- Retirement / tax model, v9.

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
  * Social Security starts in the month of the claim-age birthday (month-exact),
    is prorated in the first year, and is then received at the start of the year
    like every other cash flow.
  * The model ends at the LAST DEATH: the final simulated year is the one before
    the year of that death, and the ending balance is measured at the end of that
    year. A death date defaults to the person's horizon_age (95) birthday.
  * The household files jointly through the year of the FIRST death, then single.
    From the month of the first death, Social Security becomes the survivor's one
    check (the larger of own and survivor benefit). The deceased's accounts pass to
    the survivor and are no longer subject to the 59.5 lock.
  * Required minimum distributions (RMDs): each person must withdraw, from their OWN
    traditional balance, the prior-Dec-31 balance / IRS Uniform Lifetime divisor for the
    age they reach that year, starting at age 73 (born 1951-59) or 75 (born 1960+).
    RMDs are taken FIRST; the withdrawal plan then covers any remaining spending need,
    and an RMD larger than the need is saved to cash. Roth accounts have no RMDs.
  * Inheritances (the user's parents): each parent has a death date. At the start of the
    year of death the inherited traditional IRA is deposited into its own bucket (inh_f /
    inh_m). It must be emptied by Dec 31 of the 10th year after death: the year of death
    carries the decedent's own RMD, years 1-9 carry annual RMDs from the IRS Single Life
    Table (age in year 1, factor minus 1 each year) because the parent was already taking
    RMDs, and the whole balance is due in year 10. The withdrawal policy ("even" or "min")
    decides how much extra is taken early. Inherited IRA withdrawals are ordinary income
    with NO 10% penalty. Houses, land, belongings and cash arrive as a list of assets,
    each sold some months after the death; net proceeds go to the taxable bucket at full
    cost basis (inherited property gets a stepped-up basis, so the sale has no gain).
  * Roth conversions (optional, [roth_conversion]): in each year of a window, after the year's
    withdrawals and RMDs are known, enough of the person's OWN traditional money is converted to
    their Roth to fill the chosen federal ordinary bracket (counting the Social Security that the
    conversion itself makes taxable). The conversion is ordinary income (AGI, state tax, IRMAA
    lookback) but not spendable cash, so the plan is re-solved and the tax on it is paid from the
    usual withdrawal order (cash and taxable first). Inherited IRAs cannot be converted.
  * Asset classes (optional): each account (cash, taxable, traditional, Roth, inherited) holds a
    MIX of classes; its return is the weighted class return (constant-mix rebalancing is assumed).
    Each class pays an income yield (dividends / interest) of which a share is "qualified". Income
    in the cash and taxable accounts is taxed EVERY year (ordinary vs long-term-gains rates), is
    reinvested, and adds to the taxable account's cost basis. With no classes defined every account
    grows at the single nominal_return and nothing is taxed until sold (the older behavior).
  * Health insurance premiums are an EXTRA required expense on top of W (W is living
    spending and excludes them): marketplace coverage per person before 65, then Medicare
    Part B plus IRMAA (an income surcharge based on AGI from TWO YEARS EARLIER) from the
    month of the 65th birthday. Premiums are paid from after-tax money like any spending.
  * State (and local) income tax is part of the cost the solver sees: it is subtracted
    along with federal tax, so W is after ALL income taxes. A move to another state is
    prorated by month in the move year (the move month counts as the new state).

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
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from math import inf
from typing import Optional

import health
import rules
import statetax
import taxes

BUCKETS = ("cash", "taxable", "trad1", "trad2", "roth1", "roth2", "inh_f", "inh_m")
INH_BUCKETS = ("inh_f", "inh_m")      # inherited IRAs: father's and mother's
GROUPS = ("cash", "taxable", "traditional", "roth", "inherited")   # account groups that get an asset mix
BUCKET_GROUP = {"cash": "cash", "taxable": "taxable", "trad1": "traditional", "trad2": "traditional",
                "roth1": "roth", "roth2": "roth", "inh_f": "inherited", "inh_m": "inherited"}
CATEGORIES = ("cash", "taxable", "trad", "roth")
PENALTY_RATE = 0.10          # early-withdrawal penalty
SUCCESS_TOLERANCE = 0.01     # dollars of unfunded spending we ignore

def _default_death(birth: date, age: int) -> date:
    """The `age` birthday (day capped at 28 so Feb 29 births are safe)."""
    return date(birth.year + age, birth.month, min(birth.day, 28))


@dataclass(frozen=True)
class Asset:
    """Something inherited that is turned into cash: a house, land, belongings, a cash account.
    value is in start-year ('today's') dollars as of the parent's death, assumed to keep pace
    with inflation until received."""
    label: str
    value: float
    months_after: int = 0         # months after the death until the sale / receipt
    cost_pct: float = 0.0         # selling and estate costs as a share of the value (0.07 = 7%)

    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Asset '{self.label}': value cannot be negative.")
        if self.months_after < 0:
            raise ValueError(f"Asset '{self.label}': sale_months_after_death cannot be negative.")
        if not 0 <= self.cost_pct < 1:
            raise ValueError(f"Asset '{self.label}': sale_cost_pct must be between 0 and 1.")


@dataclass(frozen=True)
class Inheritance:
    """One inheritance event, received by person 1 when a parent dies."""
    parent: str                   # "father" or "mother"
    death_date: date
    birth_date: Optional[date] = None     # needed for the parent's own RMD in the year of death
    share: float = 1.0            # your share if siblings or others also inherit
    ira_value: float = 0.0        # traditional IRA, today's dollars at death
    assets: tuple = ()            # tuple of Asset

    def __post_init__(self):
        if self.parent not in ("father", "mother"):
            raise ValueError(f"Inheritance parent must be 'father' or 'mother', got {self.parent!r}.")
        if not 0 < self.share <= 1:
            raise ValueError(f"{self.parent}: share must be greater than 0 and at most 1.")
        if self.ira_value < 0:
            raise ValueError(f"{self.parent}: ira_value cannot be negative.")
        if self.ira_value > 0:
            if self.birth_date is None:
                raise ValueError(f"{self.parent}: birth_date is required with an IRA (it sets the parent's "
                                 f"own RMD in the year of death).")
            if self.birth_date >= self.death_date:
                raise ValueError(f"{self.parent}: death_date must be after birth_date.")
            if self.death_date.year - self.birth_date.year < rules.rmd_age(self.birth_date.year):
                raise ValueError(f"{self.parent}: the model assumes the parent was already taking RMDs at "
                                 f"death (age {rules.rmd_age(self.birth_date.year)}+); inheriting from "
                                 f"someone who died earlier follows different rules.")

    @property
    def bucket(self) -> str:
        return "inh_f" if self.parent == "father" else "inh_m"


@dataclass(frozen=True)
class AssetClass:
    """An asset class such as US stocks, bonds or cash.
    total_return: nominal return per year including income. income_yield: the part of the return paid
    out as dividends / interest each year (taxed in cash and taxable accounts); the rest is price
    growth, taxed only when sold. qualified_share: share of that income taxed at long-term-gains
    rates (qualified dividends) instead of ordinary rates (interest, other dividends)."""
    name: str
    total_return: float
    income_yield: float = 0.0
    qualified_share: float = 0.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("An asset class needs a name.")
        if self.total_return <= -1:
            raise ValueError(f"Asset class '{self.name}': total_return must be above -100%.")
        if self.income_yield < 0:
            raise ValueError(f"Asset class '{self.name}': income_yield cannot be negative.")
        if not 0 <= self.qualified_share <= 1:
            raise ValueError(f"Asset class '{self.name}': qualified_share must be between 0 and 1.")


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
    horizon_age: int               # default longevity: a person with no death date dies on this birthday
    filing_status: str             # "mfj" | "single"
    nominal_return: float
    inflation: float
    tax_enabled: bool              # False disables income tax (federal AND state) and penalties (testing)
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
    ss_claim_months1: Optional[int] = None   # claim age in months (None = at FRA); 62y0m-70y0m
    ss_claim_months2: Optional[int] = None
    ss_spousal: bool = True                  # spousal top-up (50% of the other's PIA minus your own), on by default
    person1_death: Optional[date] = None     # None = horizon_age birthday; Monte Carlo will set these
    person2_death: Optional[date] = None
    state_origin: Optional[str] = None       # state of residence now, e.g. "IN"; None = no state tax modeled
    state_dest: Optional[str] = None         # state moved to (requires move_date)
    move_date: Optional[date] = None         # None = no move
    local_tax_rate: float = 0.0              # e.g. an Indiana county income tax; applies in the starting state only
    health_enabled: bool = False             # True when [healthcare] is configured; False = no health costs modeled
    health_marketplace_monthly: float = 0.0  # per person, start-year $, covering the months before Medicare (age 65)
    health_medicare_other_monthly: float = 0.0   # per person, start-year $, ON TOP of Part B (Medigap, Part D plan...)
    health_premium_growth: float = 0.0       # annual growth of premiums ABOVE general inflation
    health_prior_magi: float = 0.0           # nominal MAGI for the two years before start_year (IRMAA lookback)
    inheritances: tuple = ()                 # tuple of Inheritance (received by person 1)
    inherited_ira_policy: str = "even"       # "even" = level draws over the window, "min" = IRS minimums then year 10
    asset_classes: tuple = ()                # tuple of AssetClass; empty = one nominal_return for everything
    allocations: tuple = ()                  # ((group, ((class name, weight), ...)), ...) one entry per account group
    conv_start: Optional[int] = None         # Roth conversion window (first year); None = no conversions
    conv_end: Optional[int] = None           # last year of the window
    conv_bracket: Optional[float] = None     # fill up to the top of this federal ordinary bracket (0.12, 0.22, ...)

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
        if self.person2_birth is None and self.ss_claim_months2 is not None:
            raise ValueError("claim_age_person2 given but there is no person2_birth.")
        for person, claim in ((1, self.ss_claim_months1), (2, self.ss_claim_months2)):
            if claim is not None and not rules.CLAIM_MIN_MONTHS <= claim <= rules.CLAIM_MAX_MONTHS:
                raise ValueError(f"Person {person}: claim age {claim // 12}y{claim % 12}m is outside "
                                 f"the allowed window of 62y0m to 70y0m.")
        if self.person2_birth is None and self.person2_death is not None:
            raise ValueError("person2_death given but there is no person2_birth.")
        for person, birth, death in ((1, self.person1_birth, self.death1),
                                     (2, self.person2_birth, self.death2)):
            if birth is None:
                continue
            if death <= birth:
                raise ValueError(f"Person {person}: death date {death} is not after birth date {birth}.")
            if death.year < self.start_year:
                raise ValueError(f"Person {person}: death date {death} is before start_year "
                                 f"{self.start_year}; the model starts with both people alive.")
        for name in ("state_origin", "state_dest"):
            if getattr(self, name) is not None:
                statetax.check_state(getattr(self, name))
        if (self.move_date is None) != (self.state_dest is None):
            raise ValueError("A move needs BOTH move_date and move_state (or neither).")
        if self.move_date is not None:
            if self.state_origin is None:
                raise ValueError("A move needs a starting state: set residence in [state].")
            if self.move_date.year < self.start_year:
                raise ValueError(f"move_date {self.move_date} is before start_year {self.start_year}.")
        if self.local_tax_rate < 0:
            raise ValueError("local_tax_rate cannot be negative.")
        if min(self.health_marketplace_monthly, self.health_medicare_other_monthly,
               self.health_prior_magi) < 0:
            raise ValueError("Health premium inputs and prior-year MAGI cannot be negative.")
        if self.health_premium_growth <= -1:
            raise ValueError("health premium growth must be greater than -100%.")
        if self.inherited_ira_policy not in ("even", "min"):
            raise ValueError("inherited_ira_policy must be 'even' or 'min'.")
        if self.asset_classes:
            names = [c.name for c in self.asset_classes]
            if len(names) != len(set(names)):
                raise ValueError("Asset class names must be unique.")
            groups = [g for g, _ in self.allocations]
            if sorted(groups) != sorted(GROUPS):
                raise ValueError(f"With asset classes defined, [allocation] needs exactly one mix for each of "
                                 f"{list(GROUPS)}; got {groups}.")
            for group, mix in self.allocations:
                total = sum(w for _, w in mix)
                if any(w < 0 for _, w in mix) or abs(total - 1.0) > 1e-6:
                    raise ValueError(f"Allocation for '{group}' must have non-negative weights adding to 1 "
                                     f"(they add to {total:g}).")
                unknown = [name for name, _ in mix if name not in names]
                if unknown:
                    raise ValueError(f"Allocation for '{group}' uses unknown asset class(es) {unknown}; "
                                     f"defined: {names}.")
        elif self.allocations:
            raise ValueError("[allocation] was given but no [[asset_class]] is defined.")
        conv = (self.conv_start, self.conv_end, self.conv_bracket)
        if any(v is not None for v in conv):
            if any(v is None for v in conv):
                raise ValueError("A Roth conversion needs start_year, end_year and target_bracket together.")
            if self.conv_start > self.conv_end:
                raise ValueError("Roth conversion start_year is after end_year.")
            _bracket_top("mfj", self.conv_bracket)          # raises if it is not a real bracket rate
        parents = [ev.parent for ev in self.inheritances]
        if len(parents) != len(set(parents)):
            raise ValueError("Each parent can appear only once in the inheritance list.")
        for ev in self.inheritances:
            if ev.death_date.year < self.start_year:
                raise ValueError(f"{ev.parent}: death_date {ev.death_date} is before start_year "
                                 f"{self.start_year}; the model starts with the inheritance still ahead.")

    # ---- derived ----
    @property
    def death1(self) -> date:
        return self.person1_death or _default_death(self.person1_birth, self.horizon_age)

    @property
    def death2(self) -> Optional[date]:
        if self.person2_birth is None:
            return None
        return self.person2_death or _default_death(self.person2_birth, self.horizon_age)

    def death_year(self, person: int) -> Optional[int]:
        d = self.death1 if person == 1 else self.death2
        return None if d is None else d.year

    @property
    def last_death_year(self) -> int:
        return max(self.death_year(i) for i in (1, 2) if self.death_year(i) is not None)

    @property
    def first_death_year(self) -> Optional[int]:
        """Year of the first death; None for a one-person household."""
        if self.person2_birth is None:
            return None
        return min(self.death_year(1), self.death_year(2))

    @property
    def n_years(self) -> int:
        return self.last_death_year - self.start_year

    def origin_months(self, year: int) -> int:
        """Months of `year` spent in the starting state. The move month counts as the new state."""
        if self.move_date is None or year < self.move_date.year:
            return 12
        if year > self.move_date.year:
            return 0
        return self.move_date.month - 1

    def filing_status_for(self, year: int) -> str:
        """Joint through the year of the first death, single afterwards."""
        if (self.filing_status == "mfj" and self.first_death_year is not None
                and year > self.first_death_year):
            return "single"
        return self.filing_status

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
            died = self.death_year(person)
            if died is not None and year > died:
                continue            # deceased: accounts pass to the survivor, penalty-free
            if self.unlock_year(person) > year:
                out.update({f"trad{person}", f"roth{person}"})
        return frozenset(out)


def _parse_claim_age(value, name):
    """[years, months] from the config -> months of age; None if not given."""
    if value is None:
        return None
    ok = (isinstance(value, (list, tuple)) and len(value) == 2
          and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
          and 0 <= value[1] <= 11)
    if not ok:
        raise ValueError(f"{name} must be [years, months], e.g. [67, 0] or [66, 6] "
                         f"(months 0-11); got {value!r}")
    return value[0] * 12 + value[1]


def _parse_death(value, name):
    """A TOML date (e.g. 2067-06-15) or None."""
    if value is None:
        return None
    if not isinstance(value, date) or hasattr(value, "hour"):
        raise ValueError(f"{name} must be a date like 2067-06-15; got {value!r}")
    return value


def _parse_inheritances(inh):
    """[inheritance.father] / [inheritance.mother] blocks -> tuple of Inheritance."""
    events = []
    for parent in ("father", "mother"):
        blk = inh.get(parent)
        if blk is None:
            continue
        if "death_date" not in blk:
            raise ValueError(f"[inheritance.{parent}] needs a death_date (remove the block for no inheritance).")
        assets = tuple(Asset(label=str(a.get("label", "asset")), value=float(a["value"]),
                             months_after=int(a.get("sale_months_after_death", 0)),
                             cost_pct=float(a.get("sale_cost_pct", 0.0)))
                       for a in blk.get("assets", []))
        events.append(Inheritance(
            parent=parent, death_date=_parse_death(blk["death_date"], f"{parent} death_date"),
            birth_date=_parse_death(blk.get("birth_date"), f"{parent} birth_date"),
            share=float(blk.get("share", 1.0)), ira_value=float(blk.get("ira_value", 0.0)), assets=assets))
    return tuple(events)


def load_params(path: str) -> Params:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    hh, m, b, s = c["household"], c["model"], c["balances"], c["spending"]
    ss = c.get("social_security", {})
    st = c.get("strategy", {})
    stt = c.get("state", {})
    hc = c.get("healthcare")                  # None = health costs not modeled
    inh = c.get("inheritance", {})
    rc = c.get("roth_conversion")             # None = no conversions
    classes = tuple(AssetClass(name=str(a["name"]), total_return=float(a["total_return"]),
                               income_yield=float(a.get("income_yield", 0.0)),
                               qualified_share=float(a.get("qualified_share", 0.0)))
                    for a in c.get("asset_class", []))
    if not classes and "nominal_return" not in m:
        raise ValueError("[model] needs nominal_return, or define [[asset_class]] entries and an [allocation].")

    if "traditional" in b or "roth" in b:
        raise ValueError(
            "v1 needs retirement balances split by person (the 59.5 rule depends on "
            "each owner's age). Replace 'traditional'/'roth' in [balances] with "
            "traditional_person1/2 and roth_person1/2 -- see config.template.toml.")
    if hh["filing_status"] not in taxes.STANDARD_DEDUCTION:
        raise ValueError("filing_status must be 'mfj' or 'single'")
    if m.get("horizon_person", "younger") != "younger":
        raise ValueError(
            "[model] horizon_person was replaced in v3 by death dates: set person1_death / "
            "person2_death in [household]. The model now ends at the last death.")

    default_order = m.get("withdrawal_order", ["cash", "taxable", "trad", "roth"])  # v0 key
    p = Params(
        start_year=int(m["start_year"]),
        person1_birth=hh["person1_birth"],
        person2_birth=hh.get("person2_birth"),
        horizon_age=int(m.get("horizon_age", 95)),
        filing_status=hh["filing_status"],
        nominal_return=float(m.get("nominal_return", 0.0)),
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
        ss_spousal=bool(ss.get("spousal_benefit", True)),
        ss_claim_months1=_parse_claim_age(ss.get("claim_age_person1"), "claim_age_person1"),
        ss_claim_months2=_parse_claim_age(ss.get("claim_age_person2"), "claim_age_person2"),
        person1_death=_parse_death(hh.get("person1_death"), "person1_death"),
        person2_death=_parse_death(hh.get("person2_death"), "person2_death"),
        state_origin=(str(stt["residence"]).upper() if stt.get("residence") else None),
        state_dest=(str(stt["move_state"]).upper() if stt.get("move_state") else None),
        move_date=_parse_death(stt.get("move_date"), "move_date"),
        local_tax_rate=float(stt.get("local_tax_rate", 0.0)),
        health_enabled=hc is not None,
        health_marketplace_monthly=float((hc or {}).get("marketplace_monthly_per_person", 0.0)),
        health_medicare_other_monthly=float((hc or {}).get("medicare_other_monthly_per_person", 0.0)),
        health_premium_growth=float((hc or {}).get("premium_growth_above_inflation", 0.0)),
        health_prior_magi=float((hc or {}).get("prior_year_magi", 0.0)),
        inheritances=_parse_inheritances(inh),
        inherited_ira_policy=inh.get("ira_withdrawal", "even"),
        conv_start=int(rc["start_year"]) if rc else None,
        conv_end=int(rc["end_year"]) if rc else None,
        conv_bracket=float(rc["target_bracket"]) if rc else None,
        asset_classes=classes,
        allocations=tuple((g, tuple((str(k), float(v)) for k, v in mix.items()))
                          for g, mix in c.get("allocation", {}).items()),
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
    lifetime_state_tax: float = 0.0
    lifetime_health: float = 0.0
    lifetime_irmaa: float = 0.0
    lifetime_inheritance: float = 0.0
    lifetime_conversions: float = 0.0

    @property
    def success(self) -> bool:
        return self.shortfall_total <= SUCCESS_TOLERANCE


def _expand(category, locked, p, want_locked):
    """Concrete buckets for a category, keeping only locked or only unlocked ones."""
    if category in ("cash", "taxable"):
        return [] if want_locked else [category]
    # inherited IRAs are traditional money that must be emptied soon, so they come before your own
    pool = (INH_BUCKETS + p.trad_order) if category == "trad" else p.roth_order
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


def _bracket_top(status, rate):
    """Top of the federal ordinary bracket with this rate, in BASE_YEAR taxable-income dollars."""
    for r, top in taxes.ORDINARY_BRACKETS[status]:
        if abs(r - rate) < 1e-9:
            return top
    rates = [r for r, _ in taxes.ORDINARY_BRACKETS[status]]
    raise ValueError(f"target_bracket {rate} is not a federal bracket rate; use one of {rates}.")


def _costs(draws, state, basis, ss_gross, scale, p, locked, state_fn=None, conv=0.0):
    """Income tax (federal + state) and early-withdrawal penalty implied by a set of draws."""
    trad_inc = draws["trad1"] + draws["trad2"] + draws["inh_f"] + draws["inh_m"]   # inherited IRA = ordinary income
    if not p.tax_enabled:
        return taxes.YearTax(0.0, 0.0, 0.0, 0.0), 0.0, trad_inc, 0.0
    gain_frac = max(0.0, 1.0 - basis / state["taxable"]) if state["taxable"] > 0 else 0.0
    gains = draws["taxable"] * gain_frac
    # dividends and interest from the cash and taxable accounts: ordinary income, and income taxed
    # at long-term-gains rates (qualified dividends), both stacked with the rest
    inv_ord, inv_qual, _ = _investment_income(p, draws, state)
    yt = taxes.year_tax(trad_inc + conv + inv_ord, gains + inv_qual, ss_gross, p.filing_status, scale)
    yt = yt._replace(inv_ordinary=inv_ord, inv_qualified=inv_qual)
    if state_fn is not None:                  # state (+ local) income tax on the same income
        yt = yt._replace(state_tax=state_fn(yt.agi, yt.taxable_ss))
    penalty = PENALTY_RATE * sum(draws[b] for b in locked)
    return yt, penalty, trad_inc, gains


def _solve_year(need, state, basis, ss_gross, scale, p, steps, locked, forced=None, state_fn=None, conv=0.0):
    """Smallest gross draw whose after-tax proceeds (plus Social Security) cover
    `need`. Net proceeds rise monotonically with the draw (marginal tax plus
    penalty stays below 100%), so bisection is safe.
    `forced` = {bucket: amount} already committed (RMDs): taken first, taxed like any
    other draw, and counted against finite caps (e.g. the standard-deduction step).
    `conv` = Roth conversion amount this year: ordinary income (and so tax) but NOT spendable cash.
    `state_fn` (from _state_tax_fn) adds state income tax to the cost of each draw, so the
    after-tax amount being solved for is after federal AND state tax.
    Returns (draws, YearTax, penalty, trad_inc, gains, surplus, shortfall)."""
    forced = forced or {}
    # what is left for the strategy to draw on after the forced draws
    avail = {b: state[b] - forced.get(b, 0.0) for b in BUCKETS}
    steps = [(bs, max(0.0, cap - sum(forced.get(b, 0.0) for b in bs)) if cap < inf else cap)
             for bs, cap in steps]
    capacity = sum(_draw_plan(inf, avail, steps).values())

    def net(x):
        plan = _draw_plan(x, avail, steps)
        draws = {b: plan[b] + forced.get(b, 0.0) for b in BUCKETS}
        yt, pen, trad_inc, gains = _costs(draws, state, basis, ss_gross, scale, p, locked, state_fn, conv)
        return (sum(draws.values()) + ss_gross - yt.income_tax - yt.state_tax - pen,
                draws, yt, pen, trad_inc, gains)

    top = net(capacity)
    if top[0] <= need:                       # cannot fund this year's spending
        return (*top[1:], 0.0, max(0.0, need - top[0]))
    base = net(0.0)
    if base[0] >= need:                      # Social Security + RMDs alone cover it
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


@lru_cache(maxsize=512)
def _bucket_rates(p):
    """{bucket: (total return, ordinary income yield, qualified income yield)}. Cached per Params.
    Without asset classes every bucket earns nominal_return with no taxable income. With them, a bucket's
    numbers are the allocation-weighted class numbers (constant-mix rebalancing)."""
    if not p.asset_classes:
        return {b: (p.nominal_return, 0.0, 0.0) for b in BUCKETS}
    classes = {c.name: c for c in p.asset_classes}
    group_rates = {}
    for group, mix in p.allocations:
        total = ordinary = qualified = 0.0
        for name, weight in mix:
            c = classes[name]
            total += weight * c.total_return
            ordinary += weight * c.income_yield * (1.0 - c.qualified_share)
            qualified += weight * c.income_yield * c.qualified_share
        group_rates[group] = (total, ordinary, qualified)
    return {b: group_rates[BUCKET_GROUP[b]] for b in BUCKETS}


def _investment_income(p, draws, state):
    """Dividends and interest earned this year in the cash and taxable accounts:
    (ordinary, qualified, part earned in the taxable account). They are earned on the balance left
    AFTER this year's draws, are reinvested, and are taxed this year. Zero without asset classes."""
    if not p.asset_classes:
        return 0.0, 0.0, 0.0
    rates = _bucket_rates(p)
    ordinary = qualified = taxable_part = 0.0
    for b in ("cash", "taxable"):
        left = state[b] - draws[b]
        o, q = left * rates[b][1], left * rates[b][2]
        ordinary += o
        qualified += q
        if b == "taxable":
            taxable_part = o + q
    return ordinary, qualified, taxable_part


def _inheritance_inflows(p, year, price):
    """What is received at the START of `year`, nominal: ({inherited IRA bucket: amount}, net cash).

    The IRA arrives in the year of the parent's death. Each asset arrives in the calendar year
    of (death month + its months_after) at value x share x (1 - selling cost). Amounts are in
    today's dollars, so they are scaled by `price`. If person 1 (the heir) has died before the
    parent's death year, nothing is received."""
    ira, cash = {}, 0.0
    for ev in p.inheritances:
        if p.death_year(1) < ev.death_date.year:
            continue
        if ev.ira_value and ev.death_date.year == year:
            ira[ev.bucket] = ira.get(ev.bucket, 0.0) + ev.ira_value * ev.share * price
        died = rules.month_index(ev.death_date.year, ev.death_date.month)
        for a in ev.assets:
            if (died + a.months_after) // 12 == year:
                cash += a.value * ev.share * price * (1.0 - a.cost_pct)
    return ira, cash


def _inherited_forced(p, year, state):
    """Withdrawals the inherited-IRA rules require this year: ({bucket: amount}, total), nominal.

      * Year of death (D): the parent's own RMD for that year (Uniform Lifetime Table, parent's age).
      * Years D+1 .. D+9: annual RMD = start-of-year balance / (Single Life factor at person 1's age
        in year D+1, minus one for each year after D+1). Required because the parent was already
        taking RMDs (IRS final regulations, enforced from 2025).
      * Year D+10: the whole balance.
    Policy "even" also takes at least balance / years-left each year (level draws); "min" takes only
    the minimums, leaving a large final-year distribution."""
    forced = {b: 0.0 for b in BUCKETS}
    for ev in p.inheritances:
        bal = state[ev.bucket]
        if not ev.ira_value or bal <= 0 or p.death_year(1) < ev.death_date.year:
            continue
        d = ev.death_date.year
        if year < d:
            continue
        if year == d:
            due = bal / rules.rmd_divisor(d - ev.birth_date.year)
        elif year < d + rules.INHERITED_IRA_YEARS:
            divisor = rules.single_life_factor(d + 1 - p.person1_birth.year) - (year - (d + 1))
            due = bal / max(divisor, 1.0)
            if p.inherited_ira_policy == "even":
                due = max(due, bal / (d + rules.INHERITED_IRA_YEARS - year + 1))
        else:
            due = bal
        forced[ev.bucket] = min(bal, due)
    return forced, sum(forced.values())


def _conversion_plan(p, status, scale, state, draws, trad_inc, gains, yt, ss_gross):
    """{own traditional bucket: amount} to convert to Roth this year, or {} for none.

    Takes this year's draws (already solved without a conversion) and finds the largest conversion
    that keeps ORDINARY TAXABLE income at or below the top of the target bracket. Taxable Social
    Security is recomputed for every trial amount, so the extra benefit taxed by the conversion is
    counted. Limited to what is left in the account after this year's draws (RMDs first); the older
    person's account is converted first. Inherited IRAs are not eligible."""
    avail = {b: max(0.0, state[b] - draws[b]) for b in p.trad_order}
    cap = sum(avail.values())
    if cap <= 0:
        return {}
    std = taxes.STANDARD_DEDUCTION[status] * scale
    top = _bracket_top(status, p.conv_bracket) * scale
    base = trad_inc + yt.inv_ordinary

    def ordinary_taxable(c):
        t = taxes.year_tax(base + c, gains + yt.inv_qualified, ss_gross, status, scale)
        return max(0.0, base + c + t.taxable_ss - std)

    if ordinary_taxable(0.0) >= top:
        return {}
    if ordinary_taxable(cap) <= top:
        amount = cap
    else:
        lo, hi = 0.0, cap
        for _ in range(60):
            mid = (lo + hi) / 2
            if ordinary_taxable(mid) <= top:
                lo = mid
            else:
                hi = mid
        amount = lo
    plan, left = {}, amount
    for b in p.trad_order:
        take = min(left, avail[b])
        if take > 0:
            plan[b] = take
            left -= take
    return plan


def _health_costs(p, year, k, status, scale, lookback_magi):
    """Health insurance premiums for `year`, nominal dollars, as a dict.

    Marketplace premium: person-months before Medicare x the monthly amount (start-year dollars).
    Medicare: person-months x (Part B standard premium + any 'other' premium you set).
    IRMAA: person-months x (Part B + Part D surcharge for the tier set by `lookback_magi`,
    the AGI from TWO years earlier, against the thresholds for this year's filing status).
    Premium amounts grow with inflation plus `health_premium_growth`; IRMAA thresholds are
    indexed by `scale` (inflation only). k = years since start_year."""
    out = {"mp_months": 0, "mc_months": 0, "marketplace": 0.0, "medicare": 0.0,
           "irmaa": 0.0, "tier": 0, "lookback": 0.0, "total": 0.0}
    if not p.health_enabled:
        return out
    for birth, death in ((p.person1_birth, p.death1), (p.person2_birth, p.death2)):
        if birth is not None:
            mp, mc = rules.health_months(birth, death, year)
            out["mp_months"] += mp
            out["mc_months"] += mc
    growth = (1 + p.inflation) * (1 + p.health_premium_growth)
    user_scale = growth ** k                        # amounts you entered, in start-year dollars
    table_scale = growth ** (year - taxes.BASE_YEAR)    # CMS table amounts, in 2026 dollars
    out["marketplace"] = out["mp_months"] * p.health_marketplace_monthly * user_scale
    out["medicare"] = out["mc_months"] * (health.PART_B_STANDARD * table_scale
                                          + p.health_medicare_other_monthly * user_scale)
    if out["mc_months"]:
        out["tier"] = health.irmaa_tier(lookback_magi, status, scale)
        part_b, part_d = health.irmaa_surcharges(out["tier"])
        out["irmaa"] = out["mc_months"] * (part_b + part_d) * table_scale
        out["lookback"] = lookback_magi
    out["total"] = out["marketplace"] + out["medicare"] + out["irmaa"]
    return out


def _state_tax_fn(p, year):
    """Returns f(agi, taxable_ss) -> state (+ local) income tax for `year`, or None when no
    state tax is modeled. Exemptions use the people on the return (alive at the start of the
    year). In the move year the tax is prorated by months in each state."""
    if p.state_origin is None:
        return None
    ages = [year - b.year for i, b in ((1, p.person1_birth), (2, p.person2_birth))
            if b is not None and p.death_year(i) >= year]
    months_origin = p.origin_months(year)

    def tax(agi, taxable_ss):
        total = 0.0
        if months_origin > 0:
            total += months_origin / 12.0 * statetax.income_tax(
                p.state_origin, year, agi, taxable_ss, ages, p.local_tax_rate)
        if months_origin < 12:
            total += (12 - months_origin) / 12.0 * statetax.income_tax(
                p.state_dest, year, agi, taxable_ss, ages)
        return total
    return tax


def _rmd_forced(p, year, state):
    """RMDs due this year: ({bucket: amount}, total), nominal dollars.

    Called with START-of-year balances, which are the prior Dec 31 balances the IRS uses.
    Each person's RMD is based on their own age (year - birth year) and their own traditional
    balance, since spouses cannot combine accounts. The year a person dies, their RMD is still
    due from their own account. Afterwards the survivor is assumed to roll the deceased's
    account into their own, so their RMD uses the combined balance (and the survivor's age).
    Roth balances never have RMDs."""
    forced = {b: 0.0 for b in BUCKETS}
    for person in (1, 2):
        birth = p.person1_birth if person == 1 else p.person2_birth
        if birth is None or year > p.death_year(person):
            continue                      # no such person, or deceased (covered by the survivor)
        age = year - birth.year
        if age < rules.rmd_age(birth.year):
            continue
        owned = [f"trad{person}"]
        other = 2 if person == 1 else 1
        if p.death_year(other) is not None and year > p.death_year(other):
            owned.append(f"trad{other}")  # spouse's account rolled over to the survivor
        amount = rules.rmd_amount(sum(state[b] for b in owned), age, rules.rmd_age(birth.year))
        for b in owned:                   # take it from the person's own account first
            take = min(amount, state[b] - forced[b])
            forced[b] += take
            amount -= take
    return forced, sum(forced.values())


def _ss_persons(p):
    """The (birth, PIA, claim months, death) list that rules.py's Social Security functions take."""
    persons = [(p.person1_birth, p.ss_fra_monthly1, p.ss_claim_months1, p.death1)]
    if p.person2_birth is not None:
        persons.append((p.person2_birth, p.ss_fra_monthly2, p.ss_claim_months2, p.death2))
    return persons


@lru_cache(maxsize=512)
def _ss_schedule(p):
    """{year: (household Social Security, of which spousal top-up)} in start-year dollars. Cached per
    Params: the solver re-simulates the same household many times, and scenario grids reuse it."""
    return rules.ss_real_by_year_detail(_ss_persons(p), p.start_year, p.start_year + p.n_years - 1,
                                        spousal=p.ss_spousal)


def _ss_income(p, year, price):
    """Total Social Security received in `year` (nominal dollars): the household schedule
    from rules.ss_real_by_year (claim ages, survivor rules) scaled to nominal."""
    return _ss_schedule(p).get(year, (0.0, 0.0))[0] * price * p.ss_multiplier


def _ss_spousal_income(p, year, price):
    """The part of this year's Social Security that is the spousal top-up (nominal dollars)."""
    return _ss_schedule(p).get(year, (0.0, 0.0))[1] * price * p.ss_multiplier


def simulate(p: Params, w_monthly_real: float, use_curve: bool = False,
             include_purchases: bool = True) -> SimResult:
    state = {"cash": p.cash, "taxable": p.taxable_value, "trad1": p.trad1,
             "trad2": p.trad2, "roth1": p.roth1, "roth2": p.roth2, "inh_f": 0.0, "inh_m": 0.0}
    basis = p.taxable_basis
    rates = _bucket_rates(p)       # per-account (return, ordinary yield, qualified yield)
    # Params copies for each filing status; each year uses the one in effect that year.
    p_by_status = {st: (p if st == p.filing_status else replace(p, filing_status=st))
                   for st in ("mfj", "single")}
    rows, penalty_years = [], []
    magi_by_year = {}          # AGI of each simulated year, for the two-year IRMAA lookback
    shortfall_total = lifetime_tax = lifetime_penalty = lifetime_state_tax = 0.0
    lifetime_health = lifetime_irmaa = lifetime_inheritance = lifetime_conversions = 0.0
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

        # joint through the year of the first death, then single
        status = p.filing_status_for(year)
        py = p_by_status[status]

        # health premiums are a required expense on top of living spending; IRMAA looks back two years
        hc = _health_costs(p, year, k, status, scale, magi_by_year.get(year - 2, p.health_prior_magi))
        living = need
        need = living + hc["total"]

        steps = _year_steps(py, locked, scale)
        ss_gross = _ss_income(p, year, price)
        ss_spousal = _ss_spousal_income(p, year, price)    # the spousal top-up inside ss_gross
        ss_spousal = _ss_spousal_income(p, year, price)

        # inheritances received at the start of this year: IRA into its own bucket, sale proceeds and
        # cash into the taxable bucket at full basis (stepped-up basis: no gain on sale)
        inh_ira_in, inh_cash_in = _inheritance_inflows(p, year, price)
        for b, amount in inh_ira_in.items():
            state[b] += amount
        if inh_cash_in:
            state["taxable"] += inh_cash_in
            basis += inh_cash_in
        lifetime_inheritance += sum(inh_ira_in.values()) + inh_cash_in

        # required minimum distributions come out first (your own, then the inherited-IRA rules)
        forced, rmd_total = _rmd_forced(p, year, state)
        inh_forced, inh_required = _inherited_forced(p, year, state)
        forced = {b: forced[b] + inh_forced[b] for b in BUCKETS}

        # state (+ local) income tax for this year, None if no state is modeled
        state_fn = _state_tax_fn(p, year)
        draws, yt, pen, trad_inc, gains, surplus, short = _solve_year(
            need, state, basis, ss_gross, scale, py, steps, locked, forced, state_fn)

        # Roth conversion: size it from the draws just solved, then re-solve so the plan pays its tax.
        # If paying that tax pulls more out of the same traditional account than is left to convert,
        # shrink the conversion to fit and solve once more.
        conv_plan = {}
        if (p.conv_bracket is not None and p.tax_enabled and p.conv_start <= year <= p.conv_end):
            conv_plan = _conversion_plan(py, status, scale, state, draws, trad_inc, gains, yt, ss_gross)
            for _ in range(2):
                if not conv_plan:
                    break
                draws, yt, pen, trad_inc, gains, surplus, short = _solve_year(
                    need, state, basis, ss_gross, scale, py, steps, locked, forced, state_fn,
                    sum(conv_plan.values()))
                fit = {b: min(a, max(0.0, state[b] - draws[b])) for b, a in conv_plan.items()}
                if all(abs(fit[b] - conv_plan[b]) < 1e-6 for b in conv_plan):
                    break
                conv_plan = {b: a for b, a in fit.items() if a > 0}
        conversion = sum(conv_plan.values())

        marginal = 0.0
        if p.tax_enabled:
            bumped = taxes.year_tax(trad_inc + conversion + yt.inv_ordinary + 1000.0, gains + yt.inv_qualified,
                                    ss_gross, status, scale)
            bumped_state = state_fn(bumped.agi, bumped.taxable_ss) if state_fn else 0.0
            marginal = ((bumped.income_tax + bumped_state) - (yt.income_tax + yt.state_tax)) / 1000.0

        if state["taxable"] > 0 and draws["taxable"] > 0:
            basis *= 1.0 - draws["taxable"] / state["taxable"]
        basis += _investment_income(p, draws, state)[2]         # reinvested dividends/interest add to basis
        for b in BUCKETS:
            state[b] -= draws[b]
        for b, amount in conv_plan.items():                     # converted money moves to the same person's Roth
            state[b] -= amount
            state[b.replace("trad", "roth")] += amount
        state["cash"] += surplus                                # unspent income is saved
        for b in BUCKETS:
            state[b] *= 1.0 + rates[b][0]                       # growth after withdrawal

        magi_by_year[year] = yt.agi
        lifetime_conversions += conversion
        lifetime_health += hc["total"]
        lifetime_irmaa += hc["irmaa"]
        shortfall_total += short
        lifetime_tax += yt.income_tax
        lifetime_state_tax += yt.state_tax
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
            "price_index": round(price, 6),     # nominal / price_index = start-year (real) dollars
            "age_person1": age1,
            "filing_status": status,
            "n_living": sum(1 for i in (1, 2) if p.death_year(i) is not None and p.death_year(i) >= year),
            "state_residence": ("" if p.state_origin is None else
                                (p.state_origin if p.origin_months(year) == 12 else p.state_dest)),
            "state_origin_months": p.origin_months(year) if p.state_origin is not None else 0,
            "spend_real": round(spend_real, 2),
            "spend_nominal": round(living, 2),
            "marketplace_person_months": hc["mp_months"],
            "medicare_person_months": hc["mc_months"],
            "health_marketplace": round(hc["marketplace"], 2),
            "health_medicare": round(hc["medicare"], 2),
            "irmaa_lookback_magi": round(hc["lookback"], 2),
            "irmaa_tier": hc["tier"],
            "health_irmaa": round(hc["irmaa"], 2),
            "health_total": round(hc["total"], 2),
            "need_nominal": round(need, 2),
            "ss_gross": round(ss_gross, 2),
            "ss_spousal": round(ss_spousal, 2),
            "ss_taxable": round(yt.taxable_ss, 2),
            "provisional_income": round(yt.provisional_income, 2),
            "agi": round(yt.agi, 2),
            "rmd_required": round(rmd_total, 2),
            "inheritance_ira_received": round(sum(inh_ira_in.values()), 2),
            "inheritance_cash_received": round(inh_cash_in, 2),
            "inherited_ira_required": round(inh_required, 2),
            "draw_cash": round(draws["cash"], 2),
            "draw_taxable": round(draws["taxable"], 2),
            "draw_trad": round(draws["trad1"] + draws["trad2"], 2),
            "draw_roth": round(draws["roth1"] + draws["roth2"], 2),
            "draw_inherited_ira": round(draws["inh_f"] + draws["inh_m"], 2),
            "gross_withdrawal": round(gross, 2),
            "realized_gains": round(gains, 2),
            "inv_income_ordinary": round(yt.inv_ordinary, 2),
            "inv_income_qualified": round(yt.inv_qualified, 2),
            "roth_conversion": round(conversion, 2),
            "tax_federal": round(yt.income_tax, 2),
            "tax_state": round(yt.state_tax, 2),
            "penalty_early_withdrawal": round(pen, 2),
            "early_withdrawal_flag": "PENALTY" if pen > 0.005 else "",
            "tax_pct_of_cash_in": round((yt.income_tax + yt.state_tax + pen) / cash_in, 4) if cash_in > 0 else 0.0,
            "marginal_ordinary_rate": round(marginal, 4),
            "surplus_to_cash": round(surplus, 2),
            "shortfall_nominal": round(short, 2),
            "end_cash": round(state["cash"], 2),
            "end_taxable": round(state["taxable"], 2),
            "end_trad1": round(state["trad1"], 2),
            "end_trad2": round(state["trad2"], 2),
            "end_roth1": round(state["roth1"], 2),
            "end_roth2": round(state["roth2"], 2),
            "end_inh_f": round(state["inh_f"], 2),
            "end_inh_m": round(state["inh_m"], 2),
            "end_total_nominal": round(end_total, 2),
            "end_total_real": round(end_total / (price * (1 + p.inflation)), 2),
        })

    final_nominal = sum(state.values())
    final_real = final_nominal / (1 + p.inflation) ** p.n_years
    return SimResult(rows, final_nominal, final_real, shortfall_total, lifetime_tax,
                     lifetime_penalty, penalty_years, depletion, lifetime_state_tax,
                     lifetime_health, lifetime_irmaa, lifetime_inheritance, lifetime_conversions)


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
