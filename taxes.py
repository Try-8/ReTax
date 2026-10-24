"""
taxes.py -- Federal tax tables and functions (v0).

All dollar figures are in BASE_YEAR (2026) dollars. The engine passes a
`scale` factor (cumulative bracket indexing) to convert them to a later year.

!! VERIFY THESE NUMBERS against IRS Rev. Proc. 2025-32 (or the current IRS
!! tables) before relying on results. Tax law is data, not code: when rules
!! change, edit the tables below and nothing else.

Scope (v1): ordinary income + long-term capital gains + taxable Social
Security (provisional-income formula), standard deduction only.
NOT yet modeled: senior deduction, extra standard deduction for age 65+, NIIT,
IRMAA, state tax, AMT, itemized deductions.
"""
from collections import namedtuple
from math import inf

BASE_YEAR = 2026

STANDARD_DEDUCTION = {"mfj": 32_200.0, "single": 16_100.0}

# (rate, upper bound of bracket) -- taxable income, BASE_YEAR dollars
ORDINARY_BRACKETS = {
    "mfj": [
        (0.10, 24_800), (0.12, 100_800), (0.22, 211_400), (0.24, 403_550),
        (0.32, 512_450), (0.35, 768_700), (0.37, inf),
    ],
    "single": [
        (0.10, 12_400), (0.12, 50_400), (0.22, 105_700), (0.24, 201_775),
        (0.32, 256_225), (0.35, 640_600), (0.37, inf),
    ],
}

# Long-term capital gains: thresholds apply to TOTAL taxable income (gains
# are stacked on top of ordinary income).
LTCG_BRACKETS = {
    "mfj": [(0.00, 98_900), (0.15, 613_700), (0.20, inf)],
    "single": [(0.00, 49_450), (0.15, 545_500), (0.20, inf)],
}


# Social Security taxation thresholds on "provisional income". Set by statute
# and NOT indexed to inflation -- compare them to NOMINAL income, never scale.
SS_TAX_THRESHOLDS = {"mfj": (32_000.0, 44_000.0), "single": (25_000.0, 34_000.0)}

YearTax = namedtuple("YearTax", "income_tax taxable_ss provisional_income agi")


def _tax_on_slice(lo, hi, brackets, scale):
    """Tax on the slice of income between lo and hi (both already in the
    same dollars as the scaled brackets)."""
    if hi <= lo:
        return 0.0
    tax, prev_top = 0.0, 0.0
    for rate, top in brackets:
        top_s = top * scale
        seg_lo, seg_hi = max(lo, prev_top), min(hi, top_s)
        if seg_hi > seg_lo:
            tax += (seg_hi - seg_lo) * rate
        prev_top = top_s
        if hi <= top_s:
            break
    return tax


def federal_tax(ordinary_income, ltcg, filing_status, scale=1.0):
    """Federal income tax (nominal dollars for the year being modeled).

    ordinary_income: e.g. traditional IRA withdrawals (nominal $)
    ltcg:            long-term gains realized (nominal $)
    scale:           bracket indexing factor relative to BASE_YEAR
    """
    std = STANDARD_DEDUCTION[filing_status] * scale
    ord_taxable = max(0.0, ordinary_income - std)
    total_taxable = max(0.0, ordinary_income + ltcg - std)
    ltcg_taxable = total_taxable - ord_taxable
    tax = _tax_on_slice(0.0, ord_taxable, ORDINARY_BRACKETS[filing_status], scale)
    tax += _tax_on_slice(ord_taxable, ord_taxable + ltcg_taxable,
                         LTCG_BRACKETS[filing_status], scale)
    return tax


def marginal_rate(ordinary_income, ltcg, filing_status, scale=1.0, step=1_000.0):
    """Approximate marginal rate on the next `step` dollars of ordinary income."""
    t0 = federal_tax(ordinary_income, ltcg, filing_status, scale)
    t1 = federal_tax(ordinary_income + step, ltcg, filing_status, scale)
    return (t1 - t0) / step


def taxable_social_security(ss_gross, other_income, filing_status):
    """Taxable portion of Social Security benefits.

    provisional income (PI) = other income + 0.5 * benefits, where other income
    is AGI excluding Social Security (plus tax-exempt interest, not modeled).
      PI <= T1:        0% taxable
      T1 < PI <= T2:   up to 50% of benefits taxable
      PI > T2:         up to 85% of benefits taxable
    Returns (taxable_amount, provisional_income). Thresholds are unindexed.
    """
    if ss_gross <= 0:
        return 0.0, other_income
    t1, t2 = SS_TAX_THRESHOLDS[filing_status]
    pi = other_income + 0.5 * ss_gross
    if pi <= t1:
        taxable = 0.0
    elif pi <= t2:
        taxable = min(0.5 * ss_gross, 0.5 * (pi - t1))
    else:
        taxable = min(0.85 * ss_gross,
                      0.85 * (pi - t2) + min(0.5 * ss_gross, 0.5 * (t2 - t1)))
    return taxable, pi


def year_tax(ordinary_income, ltcg, ss_gross, filing_status, scale=1.0):
    """Full federal calculation for one year (all nominal dollars).

    ordinary_income: traditional-account withdrawals (excluding Social Security)
    ltcg:            realized long-term gains
    ss_gross:        total Social Security benefits received
    Returns YearTax(income_tax, taxable_ss, provisional_income, agi).
    """
    other = ordinary_income + ltcg
    taxable_ss, pi = taxable_social_security(ss_gross, other, filing_status)
    tax = federal_tax(ordinary_income + taxable_ss, ltcg, filing_status, scale)
    return YearTax(tax, taxable_ss, pi, other + taxable_ss)
