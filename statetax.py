"""
statetax.py -- State (and local) income tax tables and function (v5).

One entry per state in STATES; add states by adding entries (no code changes).
A state entry describes a flat-rate income tax:

  rates                  {year: rate}. The rate in force for a year is the latest entry
                         at or before that year; the last entry continues forever
                         (the law's future changes are NOT assumed). Years before the
                         first entry use the first rate.
  exempt_social_security True if the state does not tax Social Security benefits
                         (the taxable part of the federal AGI is subtracted).
  personal_exemption     dollars per person on the return (not indexed to inflation).
  senior_exemption       extra dollars per person aged `senior_age` or older.
  senior_age             age (reached during the year) for the extra exemption.

Tax = rate x max(0, base - exemptions), where base = federal AGI as modeled, minus the
taxable part of Social Security if the state exempts it. Every state here taxes IRA
withdrawals and capital gains as ordinary income at the flat rate. A state "local" rate
(e.g. an Indiana county income tax) is added to the state rate.

!! VERIFY these against the state's revenue department before relying on results.
   Sources used (2026): Indiana HEA 1001 (2023) rate schedule 3.05% (2024), 3.00% (2025),
   2.95% (2026), 2.9% (2027+); $1,000 personal exemption per person; extra $1,000 for age
   65+ (sources conflicted between $1,000 and $1,500; $1,000 used); Social Security exempt;
   IRA withdrawals taxable. Indiana's every county also levies an income tax, handled by
   the optional `local_tax_rate` input (the county chart is published by the Indiana DOR).

NOT modeled: Indiana's small credits for age 65+ or low income, state-specific deductions
for public pensions/military pay, state AMT, state capital-gains surtaxes.
"""

STATES = {
    "IN": {
        "name": "Indiana",
        "rates": {2024: 0.0305, 2025: 0.0300, 2026: 0.0295, 2027: 0.0290},
        "exempt_social_security": True,
        "personal_exemption": 1_000.0,
        "senior_exemption": 1_000.0,
        "senior_age": 65,
    },
    "TX": {"name": "Texas", "rates": {2000: 0.0}, "exempt_social_security": True,
           "personal_exemption": 0.0, "senior_exemption": 0.0, "senior_age": 65},
    "FL": {"name": "Florida", "rates": {2000: 0.0}, "exempt_social_security": True,
           "personal_exemption": 0.0, "senior_exemption": 0.0, "senior_age": 65},
}


def check_state(code):
    if code not in STATES:
        raise ValueError(f"Unknown state '{code}'. Supported: {sorted(STATES)} "
                         f"(add more to statetax.STATES).")


def flat_rate(state, year):
    """The state's flat income tax rate in force during `year`."""
    rates = STATES[state]["rates"]
    years = sorted(rates)
    applicable = [y for y in years if y <= year]
    return rates[applicable[-1]] if applicable else rates[years[0]]


def income_tax(state, year, agi, taxable_ss, ages, local_rate=0.0):
    """Full-year state (+ local) income tax, nominal dollars.

    agi:        federal AGI as modeled (includes the taxable part of Social Security)
    taxable_ss: the taxable part of Social Security inside `agi`
    ages:       ages (reached this year) of the people on the return, for exemptions
    local_rate: local rate added to the state rate (e.g. an Indiana county tax)
    """
    s = STATES[state]
    base = agi - (taxable_ss if s["exempt_social_security"] else 0.0)
    exemptions = (len(ages) * s["personal_exemption"]
                  + sum(1 for a in ages if a >= s["senior_age"]) * s["senior_exemption"])
    return max(0.0, base - exemptions) * (flat_rate(state, year) + local_rate)
