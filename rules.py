"""
rules.py -- Age-based rules (v1): Social Security timing and the 59.5 rule.

Month arithmetic uses a "month index" = year*12 + (month-1), so we never have
to worry about leap days or end-of-month dates.

Approximations (documented in DESIGN.md):
  * Social Security benefits begin in the calendar month you reach your Full
    Retirement Age (FRA). Payment lag and the born-on-the-1st rule are ignored.
  * The 10% early-withdrawal penalty is lifted starting the calendar year you
    turn 59.5 if that happens by June 30; otherwise starting the following year.
    (Annual model: a mid-year convention avoids assuming a full year of
    penalty-free access for someone who turns 59.5 in December.)
"""
from datetime import date

EARLY_ACCESS_MONTHS = 59 * 12 + 6   # 59 years, 6 months


def month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def fra_months(birth_year: int) -> int:
    """Social Security Full Retirement Age in months, by birth year (SSA table)."""
    if birth_year <= 1937:
        return 65 * 12
    if birth_year >= 1960:
        return 67 * 12
    if birth_year <= 1942:                       # 1938-1942: 65 + 2 months per year
        return 65 * 12 + 2 * (birth_year - 1937)
    if birth_year <= 1954:                       # 1943-1954: 66
        return 66 * 12
    return 66 * 12 + 2 * (birth_year - 1954)     # 1955-1959: 66 + 2 months per year


def ss_start_index(birth: date, claim_months=None) -> int:
    """Month index of the first Social Security benefit month.
    claim_months=None means claim at FRA (the only option in v1)."""
    claim = fra_months(birth.year) if claim_months is None else claim_months
    return month_index(birth.year, birth.month) + claim


def ss_start_year_month(birth: date, claim_months=None):
    year, m0 = divmod(ss_start_index(birth, claim_months), 12)
    return year, m0 + 1


def ss_months_paid(birth: date, year: int, claim_months=None) -> int:
    """Number of benefit months falling in calendar `year` (0 to 12)."""
    start = ss_start_index(birth, claim_months)
    last_month_of_year = year * 12 + 11
    return max(0, min(12, last_month_of_year - start + 1))


def early_unlock_year(birth: date) -> int:
    """First calendar year in which retirement-account withdrawals are
    treated as penalty-free (59.5 rule, mid-year convention)."""
    idx = month_index(birth.year, birth.month) + EARLY_ACCESS_MONTHS
    year, m0 = divmod(idx, 12)
    return year if (m0 + 1) <= 6 else year + 1
