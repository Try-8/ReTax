"""
rules.py -- Age-based rules (v1): Social Security timing and the 59.5 rule.

Month arithmetic uses a "month index" = year*12 + (month-1), so we never have
to worry about leap days or end-of-month dates.

Approximations (documented in DESIGN.md):
  * Social Security benefits begin in the calendar month containing the birthday
    at which you claim (your claim age, FRA by default). The first year is
    prorated by month. Payment lag (benefits arrive the month after) and SSA's
    born-on-the-1st/2nd and "62 throughout the month" rules are ignored.
  * The 10% early-withdrawal penalty is lifted starting the calendar year you
    turn 59.5 if that happens by June 30; otherwise starting the following year.
    (Annual model: a mid-year convention avoids assuming a full year of
    penalty-free access for someone who turns 59.5 in December.)
"""
from datetime import date

EARLY_ACCESS_MONTHS = 59 * 12 + 6   # 59 years, 6 months

CLAIM_MIN_MONTHS = 62 * 12          # earliest Social Security claim age (62y0m)
CLAIM_MAX_MONTHS = 70 * 12          # delayed-retirement credits stop at 70

SURVIVOR_MIN_MONTHS = 60 * 12       # earliest age to take a survivor benefit
SURVIVOR_MAX_REDUCTION = 0.285      # reduction if taken at 60 (scales linearly to 0 at FRA)
WIDOW_LIMIT = 0.825                 # floor (share of PIA) when the deceased claimed early


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


def ss_claim_factor(birth: date, claim_months=None) -> float:
    """Fraction of the FRA benefit paid when claiming at `claim_months` of age.

    Early claim (before FRA): reduced 5/9 of 1% per month for the first 36 months
    early, then 5/12 of 1% per month beyond 36 (max 30% for FRA 67 claiming at 62).
    Delayed claim (after FRA): +2/3 of 1% per month (8%/year) up to age 70.
    The 2/3% credit applies to people born 1943 or later; earlier cohorts earned
    smaller credits and are not modeled.
    claim_months=None means claim at FRA (factor 1.0).
    """
    fra = fra_months(birth.year)
    claim = fra if claim_months is None else claim_months
    if not CLAIM_MIN_MONTHS <= claim <= CLAIM_MAX_MONTHS:
        raise ValueError(f"Claim age {claim // 12}y{claim % 12}m is outside the allowed "
                         f"window of 62y0m to 70y0m.")
    if claim >= fra:
        return 1.0 + (2.0 / 3.0) / 100.0 * (claim - fra)
    early = fra - claim
    return (1.0 - (5.0 / 9.0) / 100.0 * min(early, 36)
            - (5.0 / 12.0) / 100.0 * max(0, early - 36))


def ss_start_index(birth: date, claim_months=None) -> int:
    """Month index of the first Social Security benefit month.
    claim_months = age at claim in months (None = claim at FRA)."""
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


# ---------------------------------------------------------------------------
# Survivor benefits and the household's Social Security by year (v3)
# ---------------------------------------------------------------------------
def survivor_basis_factor(birth: date, claim_months, death_idx: int) -> float:
    """Share of the deceased's FRA benefit (PIA) that the survivor benefit is based on.

      * Deceased had already claimed: their own claim-adjusted benefit, but never
        less than 82.5% of PIA (the "widow(er) limit" when they claimed early).
      * Died before claiming, before FRA: 100% of PIA.
      * Died before claiming, after FRA: PIA plus delayed credits earned to the
        death month (capped at 70).
    """
    born = month_index(birth.year, birth.month)
    if ss_start_index(birth, claim_months) <= death_idx:
        return max(ss_claim_factor(birth, claim_months), WIDOW_LIMIT)
    fra = fra_months(birth.year)
    age_at_death = death_idx - born
    if age_at_death <= fra:
        return 1.0
    return 1.0 + (2.0 / 3.0) / 100.0 * (min(age_at_death, CLAIM_MAX_MONTHS) - fra)


def survivor_start_and_reduction(survivor_birth: date, death_idx: int):
    """(first month index, benefit factor) for a survivor benefit.
    Starts at the death month but not before the survivor is 60. Taking it before the
    survivor's FRA reduces it linearly, by 28.5% at 60. (Uses the retirement FRA
    table; SSA's survivor FRA table differs slightly for some birth years.)"""
    born = month_index(survivor_birth.year, survivor_birth.month)
    start = max(death_idx, born + SURVIVOR_MIN_MONTHS)
    fra = fra_months(survivor_birth.year)
    start_age = start - born
    if start_age >= fra:
        return start, 1.0
    return start, 1.0 - SURVIVOR_MAX_REDUCTION * (fra - start_age) / (fra - SURVIVOR_MIN_MONTHS)


SPOUSAL_FRACTION = 0.5       # spousal benefit: up to 50% of the other spouse's PIA (before reductions)


def spousal_claim_factor(birth: date, start_idx: int) -> float:
    """Share of the full spousal benefit paid when it BEGINS in month `start_idx`.

    Reduced 25/36 of 1% per month for the first 36 months before the recipient's own full
    retirement age (FRA), then 5/12 of 1% per month beyond that (FRA 67 starting at 62: 65%).
    There is NO credit for starting after FRA. The reduction depends on when the spousal benefit
    begins, which can be later than the recipient's own claim date (it waits for the other
    spouse to file)."""
    fra_idx = month_index(birth.year, birth.month) + fra_months(birth.year)
    early = max(0, fra_idx - start_idx)
    return (1.0 - (25.0 / 36.0) / 100.0 * min(early, 36)
            - (5.0 / 12.0) / 100.0 * max(0, early - 36))


def spousal_topups(persons):
    """The spousal top-up each person is entitled to while both spouses are alive.

    persons: the same (birth, PIA, claim months, death) list as ss_real_by_year. Returns one dict
    per person (all zeros unless there are exactly two people):
      excess   50% of the other spouse's PIA minus this person's own PIA, never below 0
      start    month index when it begins: the LATER of the two people's own claim months
               (the spouse must have filed, and "deemed filing" ties it to your own claim)
      factor   early-start reduction (spousal_claim_factor) for that start month
      monthly  excess x factor, in start-year dollars
    Based on the PIA, not on the other spouse's actual (possibly delayed) check, so delayed credits
    never raise it."""
    zero = {"excess": 0.0, "start": 0, "factor": 0.0, "monthly": 0.0}
    if len(persons) != 2:
        return [dict(zero) for _ in persons]
    out = []
    for me, other in ((0, 1), (1, 0)):
        birth, pia, claim, _ = persons[me]
        o_birth, o_pia, o_claim, _ = persons[other]
        excess = max(0.0, SPOUSAL_FRACTION * o_pia - pia)
        start = max(ss_start_index(birth, claim), ss_start_index(o_birth, o_claim))
        factor = spousal_claim_factor(birth, start)
        out.append({"excess": excess, "start": start, "factor": factor, "monthly": excess * factor})
    return out


def ss_real_by_year_detail(persons, first_year: int, last_year: int, spousal: bool = True) -> dict:
    """Household Social Security received in each calendar year, in start-year dollars, as
    {year: (total, spousal part of the total)}.

    persons: list of (birth date, monthly benefit at FRA (PIA), claim months or None,
             death date or None), one per person (include people whose own benefit is 0:
             they can still receive a spousal or survivor benefit).

    Rules:
      * Each person receives their own claim-adjusted benefit from their claim month
        until the month BEFORE their death month.
      * While both spouses are alive a person whose own PIA is below half the other's also
        receives the spousal top-up (spousal_topups), starting when BOTH have claimed.
        spousal=False switches it off (for comparisons and tests).
      * From the month of the first death (couples only) the household has ONE check:
        the larger of the survivor's own benefit (once claimed) and the survivor
        benefit based on the deceased. Taking the max month by month models the
        survivor switching to whichever benefit is larger. The spousal top-up ends.
      * After the second death: nothing.
    Not modeled: earnings-test effects, divorced-spouse benefits.
    """
    INF = float("inf")
    info = []
    for birth, pia, claim, death in persons:
        info.append({
            "birth": birth, "pia": pia, "claim": claim,
            "own_start": ss_start_index(birth, claim),
            "own_amt": pia * ss_claim_factor(birth, claim),
            "death_idx": INF if death is None else month_index(death.year, death.month),
            "sp_start": INF, "sp_amt": 0.0,
        })
    if spousal and len(info) == 2:
        for d, top in zip(info, spousal_topups(persons)):
            d["sp_start"], d["sp_amt"] = top["start"], top["monthly"]

    survivor_mode = len(info) == 2 and any(d["death_idx"] < INF for d in info)
    if survivor_mode:
        a, b = info
        dec, sur = (a, b) if a["death_idx"] <= b["death_idx"] else (b, a)
        first_death, second_death = dec["death_idx"], sur["death_idx"]
        surv_base = dec["pia"] * survivor_basis_factor(dec["birth"], dec["claim"], int(first_death))
        surv_start, surv_redn = survivor_start_and_reduction(sur["birth"], int(first_death))
        surv_amt = surv_base * surv_redn

    out = {}
    for year in range(first_year, last_year + 1):
        total = top_up = 0.0
        for m in range(12):
            idx = year * 12 + m
            if survivor_mode and idx >= first_death:
                if idx >= second_death:
                    continue                                   # both gone
                own = sur["own_amt"] if idx >= sur["own_start"] else 0.0
                total += max(own, surv_amt if idx >= surv_start else 0.0)
            else:
                for d in info:                                 # everyone here is alive
                    if idx < d["death_idx"] and idx >= d["own_start"]:
                        total += d["own_amt"]
                        if idx >= d["sp_start"]:
                            total += d["sp_amt"]
                            top_up += d["sp_amt"]
        out[year] = (total, top_up)
    return out


def ss_real_by_year(persons, first_year: int, last_year: int, spousal: bool = True) -> dict:
    """{year: household Social Security in start-year dollars}; see ss_real_by_year_detail."""
    return {y: total for y, (total, _) in ss_real_by_year_detail(persons, first_year, last_year, spousal).items()}


# ---------------------------------------------------------------------------
# Required minimum distributions (v4)
# ---------------------------------------------------------------------------
# IRS Uniform Lifetime Table (Pub 590-B, Appendix B, Table III), effective for
# distribution years 2022 onward. Divisor by the age you reach during the year.
_ULT = [27.4, 26.5, 25.5, 24.6, 23.7, 22.9, 22.0, 21.1, 20.2, 19.4, 18.5, 17.7, 16.8,
        16.0, 15.2, 14.4, 13.7, 12.9, 12.2, 11.5, 10.8, 10.1, 9.5, 8.9, 8.4, 7.8, 7.3,
        6.8, 6.4, 6.0, 5.6, 5.2, 4.9, 4.6, 4.3, 4.1, 3.9, 3.7, 3.5, 3.4, 3.3, 3.1, 3.0,
        2.9, 2.8, 2.7, 2.5, 2.3, 2.0]
UNIFORM_LIFETIME = dict(zip(range(72, 121), _ULT))     # ages 72..120; older ages use 2.0


def rmd_age(birth_year: int) -> int:
    """Age at which required minimum distributions begin (SECURE 2.0):
    73 if born 1951-1959, 75 if born 1960 or later. Anyone born 1950 or earlier is
    treated as 72 (the 70.5 rule for older cohorts is not modeled)."""
    if birth_year >= 1960:
        return 75
    if birth_year >= 1951:
        return 73
    return 72


def rmd_divisor(age: int) -> float:
    if age < 72:
        raise ValueError(f"The Uniform Lifetime Table starts at age 72; got {age}.")
    return UNIFORM_LIFETIME[min(age, 120)]


def rmd_amount(prior_year_end_balance: float, age: int, start_age: int) -> float:
    """RMD for the year in which the owner reaches `age`: 0 before `start_age`,
    otherwise prior-Dec-31 balance / Uniform Lifetime divisor."""
    if age < start_age or prior_year_end_balance <= 0:
        return 0.0
    return prior_year_end_balance / rmd_divisor(age)


# ---------------------------------------------------------------------------
# Health insurance months (v6)
# ---------------------------------------------------------------------------
MEDICARE_AGE_MONTHS = 65 * 12


def health_months(birth: date, death, year: int):
    """(marketplace months, Medicare months) one person has in calendar `year`.

    Marketplace coverage covers the months alive before Medicare; Medicare starts in the
    month of the 65th birthday (SSA's born-on-the-1st rule is ignored). Coverage stops the
    month BEFORE the death month, like Social Security. `death` is a date or None."""
    first, last = year * 12, year * 12 + 11
    end = last + 1 if death is None else min(month_index(death.year, death.month), last + 1)
    alive = max(0, end - first)
    medicare_start = month_index(birth.year, birth.month) + MEDICARE_AGE_MONTHS
    medicare = max(0, end - max(medicare_start, first))
    return alive - medicare, medicare


# ---------------------------------------------------------------------------
# Inherited IRAs (v7)
# ---------------------------------------------------------------------------
# IRS Single Life Expectancy Table (Pub 590-B, Appendix B, Table I; 26 CFR 1.401(a)(9)-9(b)),
# effective for distribution years 2022 onward. Ages 0..120 (120 and older use 1.0).
_SLT = [
    84.6, 83.7, 82.8, 81.8, 80.8, 79.8, 78.8, 77.9, 76.9, 75.9,          # 0-9
    74.9, 73.9, 72.9, 71.9, 70.9, 69.9, 69.0, 68.0, 67.0, 66.0,          # 10-19
    65.0, 64.1, 63.1, 62.1, 61.1, 60.2, 59.2, 58.2, 57.3, 56.3,          # 20-29
    55.3, 54.4, 53.4, 52.5, 51.5, 50.5, 49.6, 48.6, 47.7, 46.7,          # 30-39
    45.7, 44.8, 43.8, 42.9, 41.9, 41.0, 40.0, 39.0, 38.1, 37.1,          # 40-49
    36.2, 35.3, 34.3, 33.4, 32.5, 31.6, 30.6, 29.8, 28.9, 28.0,          # 50-59
    27.1, 26.2, 25.4, 24.5, 23.7, 22.9, 22.0, 21.2, 20.4, 19.6,          # 60-69
    18.8, 18.0, 17.2, 16.4, 15.6, 14.8, 14.1, 13.3, 12.6, 11.9,          # 70-79
    11.2, 10.5, 9.9, 9.3, 8.7, 8.1, 7.6, 7.1, 6.6, 6.1,                  # 80-89
    5.7, 5.3, 4.9, 4.6, 4.3, 4.0, 3.7, 3.4, 3.2, 3.0,                    # 90-99
    2.8, 2.6, 2.5, 2.3, 2.2, 2.1, 2.1, 2.1, 2.0, 2.0,                    # 100-109
    2.0, 2.0, 2.0, 1.9, 1.9, 1.8, 1.8, 1.6, 1.4, 1.1,                    # 110-119
    1.0,                                                                  # 120+
]
SINGLE_LIFE = dict(zip(range(0, 121), _SLT))

INHERITED_IRA_YEARS = 10     # SECURE Act: the account must be empty by Dec 31 of the 10th year after death


def single_life_factor(age: int) -> float:
    """Single Life Table divisor for the age (reached in the year after the owner's death)."""
    if age < 0:
        raise ValueError(f"Single Life Table age cannot be negative: {age}")
    return SINGLE_LIFE[min(age, 120)]
