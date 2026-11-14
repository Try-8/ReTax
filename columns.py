"""
columns.py -- Documentation for every column of the year-by-year CSV files (v5).

run.py writes this as `data_dictionary.txt` next to the CSVs every time it runs.

COLUMN_DOCS lists the columns IN CSV ORDER as (name, what it is, how it is calculated).
The test suite checks that this list matches the columns simulate() produces, in the same
order, so the text cannot drift from the code: add a column to model.py and the tests will
fail until it is documented here.

Numbers quoted in the text (standard deduction, thresholds, state rates) are read from the
same tables the model uses.
"""
import textwrap

import health
import statetax
import taxes
from model import BUCKET_GROUP, PENALTY_RATE, _bucket_rates

_SD = taxes.STANDARD_DEDUCTION
_T1, _T2 = taxes.SS_TAX_THRESHOLDS["mfj"]
_S1, _S2 = taxes.SS_TAX_THRESHOLDS["single"]
_IN = statetax.STATES["IN"]
_IN_RATES = ", ".join(f"{r:.2%} ({y})" for y, r in sorted(_IN["rates"].items()))
_HT = health.IRMAA_THRESHOLDS
_J = " / ".join(f"${t:,.0f}" for t in _HT["mfj"])
_S = " / ".join(f"${t:,.0f}" for t in _HT["single"])

COLUMN_DOCS = [
    ("year",
     "Calendar year being simulated.",
     "start_year + k, for k = 0, 1, 2, ... The model runs through the year BEFORE the year of "
     "the last death."),
    ("price_index",
     "Cumulative inflation factor for this year.",
     "(1 + inflation)^k, where k = years since start_year (1.0 in the first year). Nominal dollars "
     "divided by price_index = start-year ('real') dollars. End-of-year balances are deflated by "
     "price_index x (1 + inflation) instead, because they are measured a year later."),
    ("age_person1",
     "Person 1's age reached during this year.",
     "year minus person 1's birth year."),
    ("filing_status",
     "Federal filing status used this year: mfj (married filing jointly) or single.",
     "mfj through the year of the first death (the survivor may still file jointly that year), "
     "single in every later year. Always single if the config says single, and always mfj for a "
     "one-person household configured as mfj."),
    ("n_living",
     "Number of people alive at the start of the year.",
     "Count of people whose death year is this year or later."),
    ("state_residence",
     "State of residence at the end of the year (blank if no state tax is modeled).",
     "The starting state until the move year; the new state from the move year on. A move in the "
     "move year already counts as the new state here."),
    ("state_origin_months",
     "Months of this year spent in the starting state (0 to 12).",
     "12 before the move year; (move month - 1) in the move year, because the move month counts "
     "as the new state; 0 afterwards. Always 0 if no state tax is modeled. Used to prorate state tax."),
    ("spend_real",
     "After-tax spending for the year, in start-year ('real') dollars.",
     "12 x monthly spending (W) x spending-curve multiplier (1.0 in the flat run) + any one-time "
     "purchases scheduled this year. This is the same every year apart from the curve and purchases. "
     "It does NOT include health insurance premiums, which are modeled separately (health_total)."),
    ("spend_nominal",
     "The same spending in this year's ('nominal') dollars: what must actually be paid.",
     "spend_real x price_index. Compare Social Security (nominal) to this column (or to need_nominal, "
     "which adds health premiums), not to spend_real."),
    ("marketplace_person_months",
     "Person-months of marketplace (pre-Medicare) health insurance this year.",
     "For each person: the months of the year that fall before the month of their 65th birthday and "
     "before the month of their death; added across people (0 to 24 for a couple). Zero if health "
     "costs are not modeled."),
    ("medicare_person_months",
     "Person-months on Medicare this year.",
     "For each person: months from the month of their 65th birthday through the month before their "
     "death month; added across people. Zero if health costs are not modeled."),
    ("health_marketplace",
     "Marketplace insurance premiums paid this year (nominal).",
     "marketplace_person_months x the monthly premium per person you set in [healthcare] (in start-year "
     "dollars) x price_index x (1 + premium growth above inflation)^k. Treated as the full price: "
     "premium tax credits, which depend on income, are not modeled."),
    ("health_medicare",
     "Standard Medicare premiums paid this year, before any income surcharge (nominal).",
     f"medicare_person_months x (the Part B standard premium, ${health.PART_B_STANDARD:,.2f} a month in "
     f"2026 dollars, grown with inflation plus any premium growth setting + your 'other Medicare premium' "
     f"setting for Medigap, a Part D drug plan, etc.). The surcharge is in health_irmaa."),
    ("irmaa_lookback_magi",
     "The income that sets this year's Medicare surcharge: your AGI from TWO YEARS EARLIER.",
     "The agi column from year - 2. For the first two years of the model it is the prior_year_magi "
     "setting (default 0). Shown as 0 in years when nobody is on Medicare."),
    ("irmaa_tier",
     "IRMAA income tier applied this year, 0 (no surcharge) to 5.",
     f"The number of IRMAA thresholds that irmaa_lookback_magi is above, using THIS year's filing status. "
     f"2026 thresholds: joint {_J}; single {_S}. All grow with inflation except the top one, which is "
     f"fixed. After a spouse dies the survivor files single, so the couple's old joint income is compared "
     f"with the much lower single thresholds (the 'widow's trap'); the SSA-44 appeal is not modeled."),
    ("health_irmaa",
     "Income-related Medicare surcharge (IRMAA) paid this year, Parts B and D (nominal).",
     "medicare_person_months x (Part B surcharge + Part D surcharge for the tier), grown with inflation. "
     "2026 monthly amounts per person: tier 1 $81.20 + $14.50 up to tier 5 $487.00 + $91.00. Every "
     "person on Medicare pays it. Assumes everyone is enrolled in Part D."),
    ("health_total",
     "All health insurance premiums paid this year (nominal).",
     "health_marketplace + health_medicare + health_irmaa."),
    ("need_nominal",
     "Everything that had to be paid out of after-tax money this year (nominal).",
     "spend_nominal + health_total. Withdrawals plus Social Security, less taxes and penalty, must cover "
     "this; compare it, not spend_nominal alone, with your income."),
    ("ss_gross",
     "Total Social Security benefits received by the household this year (nominal).",
     "For each person: benefit at full retirement age x claim-age factor x number of benefit months "
     "this year (month-exact; the first year is prorated), plus the spousal top-up (next column) when "
     "one applies - then scaled by price_index and the benefit multiplier. After the first death the "
     "household gets ONE check: the larger of the survivor's own benefit and the survivor benefit "
     "(based on the deceased's record). Paid until the month before each person's death."),
    ("ss_spousal",
     "The part of ss_gross that is the spousal top-up (nominal). Zero unless one spouse's own benefit "
     "is under half of the other's.",
     "While both spouses are alive, the person with the smaller own benefit (at full retirement age) "
     "also receives (50% x the other's full-retirement-age benefit - their own) x a reduction. It begins "
     "when BOTH have claimed (the other spouse must have filed), so if the higher earner delays, it "
     "waits. The reduction is 25/36 of 1% per month for the first 36 months the top-up starts before the "
     "recipient's own full retirement age, then 5/12 of 1% per month (65% at 62 when FRA is 67); there is "
     "no credit for waiting. It is based on the other spouse's full-retirement-age amount, not their "
     "actual check, so their delayed-claiming credits do not raise it. It stops at the first death "
     "(the survivor benefit takes over)."),
    ("ss_taxable",
     "Part of Social Security that counts as taxable income.",
     f"IRS provisional-income formula: 0% taxable if provisional_income <= ${_T1:,.0f} (joint; "
     f"${_S1:,.0f} single); up to 50% of benefits between ${_T1:,.0f} and ${_T2:,.0f} (${_S1:,.0f} "
     f"and ${_S2:,.0f} single); up to 85% above that. These thresholds are not indexed to inflation."),
    ("provisional_income",
     "The IRS measure that decides how much Social Security is taxable.",
     "(traditional withdrawals, own and inherited, + realized_gains + dividends and interest) + 50% x ss_gross. With no Social Security it equals "
     "that other income alone, which is why it can equal agi. Zero when taxes are turned off."),
    ("agi",
     "Adjusted gross income, as the model measures it.",
     "traditional withdrawals (your own AND inherited IRAs, including RMDs) + realized_gains + inv_income_ordinary + "
     "inv_income_qualified + roth_conversion + ss_taxable. Cash and Roth draws and "
     "the cost-basis part of taxable sales are not income. Wages, interest and dividends are not "
     "modeled. Zero when taxes are turned off."),
    ("rmd_required",
     "Required minimum distribution due this year on YOUR OWN traditional accounts (nominal), both people combined. "
     "Inherited IRAs are separate: see inherited_ira_required.",
     "For each person at or past their RMD age (73 if born 1951-1959, 75 if born 1960 or later): "
     "traditional balance at the START of the year (= prior Dec 31) divided by the IRS Uniform Lifetime "
     "Table divisor for the age reached this year. Each person uses their own account; after a death the "
     "survivor uses the combined balance and their own age. Roth balances have no RMDs. It is taken "
     "first, so draw_trad is at least this large."),
    ("inheritance_ira_received",
     "Inherited IRA money that arrived this year (nominal).",
     "In the year of a parent's death: the IRA value you entered (in today's dollars) x your share x "
     "price_index. It goes into its own bucket, end_inh_f for your father's IRA or end_inh_m for your "
     "mother's, available from the start of that year."),
    ("inheritance_cash_received",
     "Cash from inherited houses, land, belongings and accounts that arrived this year, after selling costs (nominal).",
     "Each asset arrives in the calendar year of (the death month + its sale_months_after_death): value x share "
     "x price_index x (1 - sale cost). It is deposited into the taxable account at full cost basis, because "
     "inherited property gets a stepped-up basis, so selling it soon after death creates no taxable gain."),
    ("inherited_ira_required",
     "Withdrawals the inherited-IRA rules force this year (nominal), both parents' IRAs combined.",
     "Year of death: the parent's own RMD (Uniform Lifetime Table at the parent's age). Years 1-9 after death: "
     "start-of-year balance / (IRS Single Life factor at YOUR age in year 1, minus one for each later year), "
     "required because the parent was already taking RMDs. With the 'even' policy it is at least balance / "
     "years left. Year 10: the whole balance. Taken first, before spending is covered."),
    ("draw_cash",
     "Amount withdrawn from the cash bucket (nominal).",
     "Whatever the withdrawal strategy takes from cash to cover spending after Social Security, RMDs "
     "and the other buckets it is allowed to use first. Cash is not income and not taxed."),
    ("draw_taxable",
     "Total SALE PROCEEDS taken from the taxable brokerage account (nominal) - not just the gain.",
     "Set by the withdrawal strategy (for example up to one standard deduction in the "
     "taxable_std_then_cash plan). Only the gain part is income; see realized_gains."),
    ("draw_trad",
     "Total taken from your OWN traditional IRA/401(k) accounts, both people (nominal). Inherited IRAs are in "
     "draw_inherited_ira.",
     "RMDs (forced) plus any extra the strategy takes. All of it is ordinary income. Retirement accounts "
     "of anyone under 59.5 are used only as a last resort, with a penalty."),
    ("draw_roth",
     "Total taken from Roth accounts, both people (nominal).",
     "Set by the withdrawal strategy, normally used last. Not taxed, but penalized if the owner is "
     "under 59.5."),
    ("draw_inherited_ira",
     "Total taken from inherited IRAs (your father's and mother's) this year (nominal).",
     "The amount the inherited-IRA rules require (inherited_ira_required) plus anything extra the withdrawal "
     "strategy takes; inherited IRAs are used before your own traditional accounts. Ordinary income, but "
     "never subject to the 10% early-withdrawal penalty."),
    ("gross_withdrawal",
     "Total taken out of all accounts this year before any tax (nominal).",
     "draw_cash + draw_taxable + draw_trad + draw_roth + draw_inherited_ira."),
    ("realized_gains",
     "Capital gain realized by selling taxable-account assets (nominal).",
     "draw_taxable x (1 - cost basis / account value), using both figures as of the start of the year; "
     "each sale removes a proportional slice of the basis. Zero when taxes are turned off."),
    ("inv_income_ordinary",
     "Interest and other non-qualified dividends earned this year in the cash and taxable accounts (nominal).",
     "For each of those two accounts: the balance left after this year's withdrawals x the account's ordinary "
     "income yield (its asset classes' income_yield x (1 - qualified_share), weighted by the account's mix). "
     "Reinvested, and taxed as ordinary income this year whether or not you spend it. Zero unless asset "
     "classes are defined in the config."),
    ("inv_income_qualified",
     "Qualified dividends earned this year in the cash and taxable accounts (nominal).",
     "The balance left after withdrawals x the account's qualified income yield (income_yield x qualified_share, "
     "weighted by the mix). Taxed at the long-term capital gains rates (0/15/20%), stacked on top of your "
     "ordinary income, in the year it is earned; reinvested, so it adds to the taxable account's cost basis. "
     "Zero unless asset classes are defined."),
    ("roth_conversion",
     "Amount moved from your own traditional IRAs to your Roth accounts this year (nominal).",
     "Only in the years of the [roth_conversion] window. Each year, after the year's withdrawals and RMDs are "
     "known, the largest amount that keeps ORDINARY TAXABLE income at or below the top of the target federal "
     "bracket (top of bracket + standard deduction, less other ordinary income). The Social Security that the "
     "conversion itself makes taxable is counted. Limited to what is left in the account after the year's "
     "draws; the older person's account goes first; inherited IRAs are never converted. It is ordinary income "
     "(it is inside agi, so it raises state tax and the IRMAA two years later) but it is not spendable cash: "
     "the tax on it is paid by the normal withdrawal order (cash and taxable first). It lands in the same "
     "person's Roth (end_roth1 / end_roth2). Needs taxes turned on."),
    ("tax_federal",
     "Federal income tax for the year (nominal).",
     f"Ordinary income = traditional draws (own and inherited) + inv_income_ordinary + ss_taxable, less the standard deduction "
     f"(${_SD['mfj']:,.0f} joint / ${_SD['single']:,.0f} single in 2026 dollars), taxed at the ordinary "
     f"brackets. realized_gains and inv_income_qualified are stacked on top and taxed at 0% / 15% / 20%. Standard deduction and "
     f"brackets are indexed by inflation from 2026. Not modeled: NIIT, IRMAA, itemized deductions, the "
     f"extra deduction for age 65+. Zero when taxes are turned off."),
    ("tax_state",
     "State (plus optional local/county) income tax for the year (nominal).",
     f"For each state lived in this year: rate x (agi - taxable Social Security [if the state exempts "
     f"it] - exemptions), times months in that state / 12. Indiana: flat rate {_IN_RATES}, then 2.9%; "
     f"exemptions of ${_IN['personal_exemption']:,.0f} per person plus ${_IN['senior_exemption']:,.0f} "
     f"for each person 65+; Social Security exempt; IRA withdrawals and capital gains taxed at the flat "
     f"rate. Texas and Florida: none. The optional local_tax_rate (e.g. an Indiana county tax) is added "
     f"to the starting state's rate and stops when you move. Zero if no state is configured or taxes "
     f"are off."),
    ("penalty_early_withdrawal",
     f"{PENALTY_RATE:.0%} early-withdrawal penalty on retirement money taken before 59.5 (nominal).",
     f"{PENALTY_RATE:.0%} x draws from the traditional and Roth accounts of anyone not yet 59.5 (the "
     f"model applies it to Roth draws too, which is conservative). Such accounts are touched only if "
     f"all other money is gone. The deceased's accounts and inherited IRAs are never penalized. 59.5 is "
     f"reached in the calendar year you turn 59.5 if that is by June 30, otherwise the next year."),
    ("early_withdrawal_flag",
     "The word PENALTY in any year a penalty was paid; blank otherwise.",
     "Shown when penalty_early_withdrawal is greater than zero."),
    ("tax_pct_of_cash_in",
     "Total taxes and penalty as a share of the money that came in (a rough effective rate).",
     "(tax_federal + tax_state + penalty_early_withdrawal) / (gross_withdrawal + ss_gross)."),
    ("marginal_ordinary_rate",
     "Federal + state tax on the NEXT dollar of traditional-account income.",
     "(tax with $1,000 more traditional income - tax now) / $1,000, federal and state together. "
     "Includes the effect that extra income makes more Social Security taxable, so it can exceed the "
     "bracket rate."),
    ("surplus_to_cash",
     "Money left over after spending and taxes, deposited into the cash bucket (nominal).",
     "Happens when Social Security plus forced RMDs cover spending: "
     "(draws + ss_gross - taxes - penalty) - need_nominal, when that is positive; otherwise 0."),
    ("shortfall_nominal",
     "Spending that could NOT be paid because the money ran out (nominal). 0 = fully funded.",
     "need_nominal - (draws + ss_gross - taxes - penalty), when positive. The money 'runs out' in "
     "the first year this is above $0.01. The W solver searches for the highest spending that keeps "
     "this at zero in every year."),
    ("end_cash",
     "Cash balance at the end of the year (nominal).",
     "(start balance - draw_cash + surplus_to_cash) x (1 + the cash account's return): its asset mix's "
     "weighted return, or the single nominal return if no asset classes are defined. Its interest is "
     "taxed each year (inv_income_ordinary)."),
    ("end_taxable",
     "Taxable brokerage balance at the end of the year (nominal).",
     "(start balance - draw_taxable) x (1 + the taxable account's return). Reinvested dividends and "
     "interest add to its cost basis, which is why gains build only from price growth."),
    ("end_trad1",
     "Person 1's traditional account balance at the end of the year (nominal).",
     "(start balance - that account's draws) x (1 + that account's return, from its asset mix)."),
    ("end_trad2",
     "Person 2's traditional account balance at the end of the year (nominal).",
     "Same as end_trad1 for person 2. After a death the deceased's account stays in its own column but "
     "is usable by the survivor."),
    ("end_roth1",
     "Person 1's Roth balance at the end of the year (nominal).",
     "(start balance - that account's draws) x (1 + that account's return, from its asset mix)."),
    ("end_roth2",
     "Person 2's Roth balance at the end of the year (nominal).",
     "Same as end_roth1 for person 2."),
    ("end_inh_f",
     "Inherited IRA from your father: balance at the end of the year (nominal).",
     "(start balance + any deposit this year - draws) x (1 + that account's return). It must reach zero by December 31 "
     "of the 10th year after his death."),
    ("end_inh_m",
     "Inherited IRA from your mother: balance at the end of the year (nominal).",
     "Same as end_inh_f for your mother."),
    ("end_total_nominal",
     "Total of all eight end-of-year balances (nominal).",
     "end_cash + end_taxable + end_trad1 + end_trad2 + end_roth1 + end_roth2 + end_inh_f + end_inh_m."),
    ("end_total_real",
     "The same total in start-year ('real') dollars.",
     "end_total_nominal / (price_index x (1 + inflation))."),
]


def _wrap(label, text, width=92):
    return textwrap.fill(text, width=width, initial_indent=f"  {label} ",
                         subsequent_indent="        ")


def run_settings(p):
    """A short summary of the settings behind this run's numbers."""
    lines = [
        f"Years simulated: {p.start_year}-{p.start_year + p.n_years - 1} ({p.n_years} years); "
        f"the plan ends at the last death ({p.last_death_year}).",
        (f"Inflation {p.inflation:.2%}; returns by account (asset classes, dividends and interest taxed each "
         f"year in cash and taxable): " + "; ".join(
             f"{g} {_bucket_rates(p)[b][0]:.2%} (income {_bucket_rates(p)[b][1] + _bucket_rates(p)[b][2]:.2%})"
             for g, b in (("cash", "cash"), ("taxable", "taxable"), ("traditional", "trad1"),
                          ("roth", "roth1"), ("inherited", "inh_f"))) + ".")
        if p.asset_classes else
        f"Inflation {p.inflation:.2%}, nominal return {p.nominal_return:.2%} (one rate for all accounts, "
        f"nothing taxed until sold).",
        f"Filing status: {p.filing_status}"
        + (" (single after the first death)" if p.filing_status == "mfj" and p.person2_birth else "") + ".",
        f"Withdrawal strategy: {p.strategy}.",
    ]
    if p.state_origin is None:
        lines.append("State income tax: none modeled.")
    else:
        text = f"State income tax: {p.state_origin}"
        if p.local_tax_rate:
            text += f" plus local rate {p.local_tax_rate:.2%}"
        if p.move_date is not None:
            text += f"; moves to {p.state_dest} in {p.move_date:%b %Y}"
        lines.append(text + ".")
    for n, claim, amt in ((1, p.ss_claim_months1, p.ss_fra_monthly1), (2, p.ss_claim_months2, p.ss_fra_monthly2)):
        if amt > 0:
            when = "at full retirement age" if claim is None else f"at {claim // 12}y{claim % 12}m"
            lines.append(f"Social Security, person {n}: ${amt:,.0f}/month (today's $) at full retirement "
                         f"age, claimed {when}.")
    if not p.health_enabled:
        lines.append("Health insurance costs: not modeled.")
    else:
        extra = (f", plus ${p.health_medicare_other_monthly:,.0f}/month other Medicare premiums"
                 if p.health_medicare_other_monthly else "")
        growth = (f"; premiums grow {p.health_premium_growth:.1%} a year above inflation"
                  if p.health_premium_growth else "")
        lines.append(f"Health insurance: ${p.health_marketplace_monthly:,.0f}/month per person (today's $) "
                     f"before 65; from 65 Medicare Part B{extra} plus IRMAA on a two-year lookback{growth}. "
                     f"These are paid on top of living spending (W).")
    for ev in p.inheritances:
        what = []
        if ev.ira_value:
            what.append(f"IRA ${ev.ira_value:,.0f}")
        what += [f"{a.label} ${a.value:,.0f}" for a in ev.assets]
        lines.append(f"Inheritance from your {ev.parent}, dying {ev.death_date:%b %Y}: "
                     f"{', '.join(what) or 'nothing listed'} (today's dollars; your share {ev.share:.0%}). "
                     f"Inherited IRA drawn '{p.inherited_ira_policy}'.")
    lines.append(f"Death dates: person 1 {p.death1:%b %Y}"
                 + (f", person 2 {p.death2:%b %Y}" if p.death2 else "") + ".")
    return lines


def render_data_dictionary(p=None, version="v5"):
    """The full text of data_dictionary.txt."""
    out = [
        f"DATA DICTIONARY -- retirement model {version}",
        "=" * 70,
        "Describes every column in flat_path.csv and curve_path.csv. Both files have the same",
        "columns, one row per simulated year. This file is regenerated each time run.py runs.",
        "",
    ]
    if p is not None:
        out += ["THIS RUN", "-" * 70] + [textwrap.fill(f"* {line}", 92, subsequent_indent="  ")
                                          for line in run_settings(p)] + [""]
    out += [
        "HOW TO READ THE NUMBERS",
        "-" * 70,
        textwrap.fill("* Dollar amounts are NOMINAL (that year's dollars) unless a column says real. "
                      "To compare with start-year dollars, divide by price_index.", 92, subsequent_indent="  "),
        textwrap.fill("* Timing: within each year, withdrawals and Social Security are treated as arriving "
                      "at the START of the year, then balances grow for the whole year. Balances "
                      "named end_* are measured after that growth.", 92, subsequent_indent="  "),
        textwrap.fill("* Money amounts are rounded to cents in the CSV, so small sums can differ by a "
                      "cent or two.", 92, subsequent_indent="  "),
        textwrap.fill("* 'W' is the monthly spending level being simulated; spend_real = 12 x W (x curve) "
                      "+ purchases.", 92, subsequent_indent="  "),
        "",
        "COLUMNS (in the same order as the CSV)",
        "-" * 70,
    ]
    for name, what, how in COLUMN_DOCS:
        out.append(name)
        out.append(_wrap("What:", what))
        out.append(_wrap("How: ", how))
        out.append("")
    return "\n".join(out)
