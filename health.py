"""
health.py -- Medicare Part B / Part D premium and IRMAA tables (v6).

All dollar amounts are MONTHLY, PER PERSON, in 2026 dollars (taxes.BASE_YEAR).
Sources: CMS 2026 Medicare Parts A & B Premiums and Deductibles fact sheet (announced
Nov 2025) and SSA POMS HI 01101.020 sliding-scale tables, as summarized by several 2026
IRMAA guides (the sources agreed). !! Verify against medicare.gov / CMS before relying on
results; CMS publishes the next year's figures each fall.

IRMAA (Income-Related Monthly Adjustment Amount):
  * A surcharge added to the Part B premium AND to the Part D drug-plan premium.
  * Based on MAGI from TWO YEARS EARLIER (2026 premiums use 2024 income).
  * Each person on Medicare pays it individually (a couple pays it twice).
  * Tier thresholds differ by filing status. The status used is the one in effect in the
    premium year (so a survivor's premium can be set by the couple's old joint income
    measured against the single thresholds -- the "widow's trap").
  * Tier = how many thresholds MAGI is ABOVE (at or below the first threshold = standard).
  * Thresholds are indexed to inflation each year, EXCEPT the top threshold, which has been
    flat ($500k single / $750k joint since 2019) -- treated as fixed here.

NOT modeled: Form SSA-44 life-changing-event appeals (they help when income truly fell but
not when only the filing status changed), the Part B late-enrollment penalty, Medicare
Advantage, Medigap, Part A premiums, deductibles and out-of-pocket costs.
"""

PART_B_STANDARD = 202.90                     # 2026 standard Part B premium per month

# MAGI thresholds (2026 dollars). Tier n applies when MAGI is above the n-th threshold.
IRMAA_THRESHOLDS = {
    "mfj":    [218_000.0, 274_000.0, 342_000.0, 410_000.0, 750_000.0],
    "single": [109_000.0, 137_000.0, 171_000.0, 205_000.0, 500_000.0],
}

# Monthly surcharges per person for tiers 1..5 (total Part B premium minus the standard
# $202.90: totals are 284.10, 405.80, 527.50, 649.20, 689.90).
PART_B_SURCHARGE = [81.20, 202.90, 324.60, 446.30, 487.00]
PART_D_SURCHARGE = [14.50, 37.50, 60.40, 83.30, 91.00]


def irmaa_tier(magi, status, scale=1.0):
    """IRMAA tier 0-5 for a MAGI. `scale` indexes every threshold except the top one."""
    thresholds = IRMAA_THRESHOLDS[status]
    tier = 0
    for n, limit in enumerate(thresholds, start=1):
        if magi > (limit if n == len(thresholds) else limit * scale):
            tier = n
    return tier


def irmaa_surcharges(tier):
    """(Part B surcharge, Part D surcharge) per person per month, 2026 dollars."""
    if tier <= 0:
        return 0.0, 0.0
    return PART_B_SURCHARGE[tier - 1], PART_D_SURCHARGE[tier - 1]
