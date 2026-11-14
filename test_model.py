"""
Tests for the retirement model.

Run from this folder:
    python -m unittest                      # fast loop (slow tests are skipped)
    SLOW_TESTS=1 python -m unittest -v      # everything, before you trust a change
    python -m unittest --durations 10       # see which tests are slow (Python 3.12+)
    UPDATE_GOLDEN=1 python -m unittest      # re-record the golden regression file

When to update the golden file: ONLY after a change that you intend to shift
the numbers (e.g. you corrected a tax table). Review the diff first -- an
unexpected golden failure is the test doing its job.

Notes on tolerances
  * Year rows returned by simulate() are rounded to cents, so row-level
    accounting identities hold to ~0.03, not 1e-6. The test
    `test_year_level_identity_to_float_precision` checks the identity at full
    float precision by calling the engine's per-year solver directly.
"""
import dataclasses
import json
import os
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import date

import columns
import health
import model
import rules
import statetax
import taxes
from model import Asset, AssetClass, Inheritance, Params, simulate, solve_max_w

slow = unittest.skipUnless(os.getenv("SLOW_TESTS"), "slow test: set SLOW_TESTS=1 to run")
ROW_CENT_TOL = 0.03   # rows are rounded to cents; up to ~5 rounded terms are summed


def make_params(**overrides):
    """Build Params from an EXPLICIT field list. Every field is spelled out here
    on purpose: if a new field is added to Params, this helper fails loudly until
    it is added below, so tests never silently inherit a new default."""
    fields = {
        "start_year": 2027,
        "person1_birth": date(1965, 1, 1),
        "person2_birth": None,
        "horizon_age": 95,
        "filing_status": "mfj",
        "nominal_return": 0.05,
        "inflation": 0.025,
        "tax_enabled": True,
        "cash": 0.0,
        "taxable_value": 0.0,
        "taxable_basis": 0.0,
        "trad1": 0.0,
        "trad2": 0.0,
        "roth1": 0.0,
        "roth2": 0.0,
        "target_monthly_real": 0.0,
        "curve": ((0.0, 1.0),),
        "purchases": (),
        "strategy": "fixed_order",
        "fixed_order": ("cash", "taxable", "trad", "roth"),
        "early_access_order": ("roth", "trad"),
        "ss_fra_monthly1": 0.0,
        "ss_fra_monthly2": 0.0,
        "ss_multiplier": 1.0,
        "ss_claim_months1": None,
        "ss_claim_months2": None,
        "ss_spousal": True,
        "person1_death": None,
        "person2_death": None,
        "state_origin": None,
        "state_dest": None,
        "move_date": None,
        "local_tax_rate": 0.0,
        "health_enabled": False,
        "health_marketplace_monthly": 0.0,
        "health_medicare_other_monthly": 0.0,
        "health_premium_growth": 0.0,
        "health_prior_magi": 0.0,
        "inheritances": (),
        "inherited_ira_policy": "even",
        "asset_classes": (),
        "allocations": (),
        "conv_start": None,
        "conv_end": None,
        "conv_bracket": None,
    }
    real = {f.name for f in dataclasses.fields(Params)}
    missing, unknown = real - set(fields), set(fields) - real
    assert not missing, f"make_params is missing Params fields: {sorted(missing)}"
    assert not unknown, f"make_params lists fields Params no longer has: {sorted(unknown)}"
    bad = set(overrides) - real
    assert not bad, f"unknown override(s): {sorted(bad)}"
    fields.update(overrides)
    return Params(**fields)


# ===========================================================================
# Tax functions
# ===========================================================================
class TaxTests(unittest.TestCase):
    def test_ordinary_bracket_math(self):
        tax = taxes.federal_tax(32_200 + 100_000, 0, "mfj")
        self.assertAlmostEqual(tax, 0.10 * 24_800 + 0.12 * (100_000 - 24_800), places=4)

    def test_zero_tax_under_standard_deduction(self):
        self.assertEqual(taxes.federal_tax(30_000, 0, "mfj"), 0.0)

    def test_gains_in_zero_bracket(self):
        self.assertEqual(taxes.federal_tax(0, 100_000, "mfj"), 0.0)

    def test_gains_stack_on_ordinary(self):
        ordinary = 32_200 + 98_900
        self.assertAlmostEqual(
            taxes.federal_tax(ordinary, 10_000, "mfj"),
            taxes.federal_tax(ordinary, 0, "mfj") + 1_500.0, places=4)

    def test_indexing_scales_thresholds(self):
        self.assertAlmostEqual(taxes.federal_tax(2 * 32_200 + 2 * 50_000, 0, "mfj", scale=2.0),
                               2 * taxes.federal_tax(32_200 + 50_000, 0, "mfj"), places=4)


class SocialSecurityTaxTests(unittest.TestCase):
    def test_well_below_first_threshold(self):
        # PI = 10,000 + 10,000 = 20,000 < 32,000
        t, pi = taxes.taxable_social_security(20_000, 10_000, "mfj")
        self.assertEqual(t, 0.0)
        self.assertAlmostEqual(pi, 20_000.0)

    def test_at_first_threshold(self):
        # PI = 20,000 + 12,000 = 32,000: exactly at T1, nothing taxable
        self.assertEqual(taxes.taxable_social_security(24_000, 20_000, "mfj")[0], 0.0)

    def test_fifty_percent_tier(self):
        # PI = 25,000 + 12,000 = 37,000 -> 50% of the 5,000 above 32,000
        t, pi = taxes.taxable_social_security(24_000, 25_000, "mfj")
        self.assertAlmostEqual(t, 2_500.0)
        self.assertAlmostEqual(pi, 37_000.0)

    def test_eighty_five_percent_tier(self):
        # PI = 30,000 + 18,000 = 48,000 -> 0.85*4,000 + 6,000 = 9,400
        self.assertAlmostEqual(taxes.taxable_social_security(36_000, 30_000, "mfj")[0], 9_400.0)

    def test_cap_at_85_percent(self):
        self.assertAlmostEqual(taxes.taxable_social_security(30_000, 100_000, "mfj")[0], 25_500.0)

    def test_single_thresholds(self):
        # PI = 30,000 + 10,000 = 40,000 single (T1 25k, T2 34k):
        # 0.85*(40,000-34,000) + min(0.5*20,000, 0.5*9,000) = 5,100 + 4,500
        self.assertAlmostEqual(taxes.taxable_social_security(20_000, 30_000, "single")[0],
                               0.85 * 6_000 + 4_500.0)

    def test_thresholds_not_indexed_by_scale(self):
        a = taxes.year_tax(30_000, 0, 36_000, "mfj", scale=1.0)
        b = taxes.year_tax(30_000, 0, 36_000, "mfj", scale=2.0)
        self.assertAlmostEqual(a.taxable_ss, b.taxable_ss)


class RulesTests(unittest.TestCase):
    def test_fra_table(self):
        self.assertEqual(rules.fra_months(1955), 66 * 12 + 2)
        self.assertEqual(rules.fra_months(1959), 66 * 12 + 10)
        self.assertEqual(rules.fra_months(1960), 67 * 12)
        self.assertEqual(rules.fra_months(1950), 66 * 12)

    def test_months_paid(self):
        b = date(1965, 6, 15)            # FRA 67 -> June 2032
        self.assertEqual(rules.ss_start_year_month(b), (2032, 6))
        self.assertEqual(rules.ss_months_paid(b, 2031), 0)
        self.assertEqual(rules.ss_months_paid(b, 2032), 7)   # Jun..Dec
        self.assertEqual(rules.ss_months_paid(b, 2033), 12)

    def test_early_unlock_year(self):
        # 59.5 on/before June 30 -> unlocked that calendar year; later -> next year
        self.assertEqual(rules.early_unlock_year(date(1965, 12, 1)), 2025)  # 59.5 = Jun 2025
        self.assertEqual(rules.early_unlock_year(date(1966, 1, 1)), 2026)   # 59.5 = Jul 2025
        self.assertEqual(rules.early_unlock_year(date(1966, 2, 1)), 2026)   # 59.5 = Aug 2025
        self.assertEqual(rules.early_unlock_year(date(1965, 6, 30)), 2025)  # 59.5 = Dec 2024 -> next year


# ===========================================================================
# Params: validation, horizon, ordering
# ===========================================================================
class ParamsTests(unittest.TestCase):
    def test_make_params_default_builds(self):
        self.assertEqual(make_params().n_years, 1965 + 95 - 2027)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            make_params(strategy="nope")

    def test_bad_fixed_order_category_rejected(self):
        with self.assertRaises(ValueError):
            make_params(fixed_order=("cash", "bogus"))

    def test_bad_early_access_order_category_rejected(self):
        with self.assertRaises(ValueError):
            make_params(early_access_order=("cash",))      # only trad/roth allowed

    def test_person2_money_without_person2_birth_rejected(self):
        for field in ("trad2", "roth2", "ss_fra_monthly2"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    make_params(**{field: 1.0}, person2_birth=None)

    def test_default_deaths_are_horizon_age_birthdays(self):
        p = make_params(person2_birth=date(1970, 3, 9))
        self.assertEqual(p.death1, date(2060, 1, 1))
        self.assertEqual(p.death2, date(2065, 3, 9))

    def test_default_death_is_safe_for_feb_29_birthdays(self):
        p = make_params(person1_birth=date(1972, 2, 29))
        self.assertEqual(p.death1, date(2067, 2, 28))

    def test_horizon_is_the_last_death(self):
        p = make_params(person2_birth=date(1970, 1, 1), person1_death=date(2050, 5, 1),
                        person2_death=date(2058, 8, 1))
        self.assertEqual((p.first_death_year, p.last_death_year), (2050, 2058))
        self.assertEqual(p.n_years, 2058 - 2027)

    def test_default_horizon_is_the_younger_persons_95th_birthday(self):
        # same result the old horizon_person="younger" setting gave
        p = make_params(person2_birth=date(1970, 1, 1))
        self.assertEqual(p.n_years, 1970 + 95 - 2027)

    def test_single_person_household(self):
        p = make_params()
        self.assertIsNone(p.death2)
        self.assertIsNone(p.first_death_year)
        self.assertEqual(p.last_death_year, 2060)

    def test_bad_death_dates_rejected(self):
        with self.assertRaises(ValueError):                       # before birth
            make_params(person1_death=date(1960, 1, 1))
        with self.assertRaises(ValueError):                       # before the model starts
            make_params(person1_death=date(2026, 12, 31))
        with self.assertRaises(ValueError):                       # no person 2
            make_params(person2_death=date(2050, 1, 1))

    def test_death_in_first_simulated_year_allowed(self):
        p = make_params(person2_birth=date(1970, 1, 1), person1_death=date(2027, 3, 1))
        self.assertEqual(p.first_death_year, 2027)

    def test_older_person_drawn_first(self):
        older2 = make_params(person2_birth=date(1960, 1, 1))
        self.assertEqual(older2.trad_order, ("trad2", "trad1"))
        self.assertEqual(older2.roth_order, ("roth2", "roth1"))
        older1 = make_params(person2_birth=date(1970, 1, 1))
        self.assertEqual(older1.trad_order, ("trad1", "trad2"))


class CurveTests(unittest.TestCase):
    def test_curve_multiplier_interpolation(self):
        c = ((0, 1.0), (100, 0.5))
        self.assertAlmostEqual(model.curve_multiplier(c, 50), 0.75)
        self.assertAlmostEqual(model.curve_multiplier(c, -10), 1.0)    # flat below range
        self.assertAlmostEqual(model.curve_multiplier(c, 200), 0.5)    # flat above range

    def test_curve_points_may_be_unsorted(self):
        self.assertAlmostEqual(model.curve_multiplier(((100, 0.5), (0, 1.0)), 50), 0.75)

    def test_single_point_curve_is_flat(self):
        self.assertAlmostEqual(model.curve_multiplier(((0.0, 1.0),), 80), 1.0)

    def test_curve_reduces_late_spending_so_w_scale_rises(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",),
                        curve=((0, 1.0), (95, 0.7)))
        self.assertGreater(solve_max_w(p, use_curve=True), solve_max_w(p, use_curve=False))

    def test_use_curve_changes_spending_rows(self):
        p = make_params(roth1=1_000_000.0, curve=((0, 1.0), (95, 0.7)))
        flat = simulate(p, 5_000.0, use_curve=False).rows
        curved = simulate(p, 5_000.0, use_curve=True).rows
        self.assertAlmostEqual(flat[-1]["spend_real"], 60_000.0)
        self.assertLess(curved[-1]["spend_real"], flat[-1]["spend_real"])


# ===========================================================================
# Engine basics
# ===========================================================================
class EngineTests(unittest.TestCase):
    def test_zero_everything_gives_simple_division(self):
        # Strategy and order pinned explicitly so a default change cannot silently
        # change what this closed-form test measures.
        p = make_params(nominal_return=0.0, inflation=0.0, tax_enabled=False, trad1=1_200_000.0,
                        strategy="fixed_order", fixed_order=("trad",))
        self.assertAlmostEqual(solve_max_w(p), 1_200_000.0 / (12 * p.n_years), delta=0.05)

    def test_all_roth_matches_annuity_formula(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",))
        rr = (1 + p.nominal_return) / (1 + p.inflation) - 1
        annual = 1_000_000.0 / sum((1 + rr) ** (-k) for k in range(p.n_years))
        self.assertAlmostEqual(solve_max_w(p), annual / 12, delta=0.05)

    def test_after_tax_spending_equals_need(self):
        p = make_params(trad1=2_000_000.0, ss_fra_monthly1=3_000.0)
        res = simulate(p, 6_000.0)
        self.assertTrue(res.success)
        for r in res.rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)

    def test_solved_w_is_the_edge(self):
        p = make_params(trad1=900_000.0, roth1=200_000.0, taxable_value=300_000.0,
                        taxable_basis=200_000.0, cash=50_000.0, ss_fra_monthly1=2_500.0)
        w = solve_max_w(p)
        res = simulate(p, w)
        self.assertTrue(res.success)
        self.assertFalse(simulate(p, w + 5.0).success)
        # "about zero left", scale-invariant; >= 0 (not > 0) because feasibility
        # allows up to SUCCESS_TOLERANCE of rounding at the edge.
        self.assertGreaterEqual(res.final_nominal, 0.0)
        self.assertLess(res.final_nominal, 0.05 * p.total_start_balance)

    def test_taxes_reduce_w(self):
        p_tax = make_params(trad1=1_500_000.0)
        self.assertLess(solve_max_w(p_tax), solve_max_w(replace(p_tax, tax_enabled=False)))

    def test_cost_basis_reduces_tax(self):
        # Large enough that gains exceed the 0% long-term-gains bracket.
        hi = make_params(taxable_value=6_000_000.0, taxable_basis=6_000_000.0)
        lo = make_params(taxable_value=6_000_000.0, taxable_basis=1_000_000.0)
        self.assertGreater(solve_max_w(hi), solve_max_w(lo))
        # Mechanism: same spending, first year -- the low-basis case pays capital-gains tax.
        r_hi, r_lo = simulate(hi, 30_000.0).rows[0], simulate(lo, 30_000.0).rows[0]
        self.assertEqual(r_hi["tax_federal"], 0.0)
        self.assertGreater(r_lo["tax_federal"], 0.0)
        self.assertLess(r_hi["tax_federal"], r_lo["tax_federal"])

    # ---- withdrawal order ----
    def test_fixed_order_draws_in_requested_order(self):
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0)
        roth_first = simulate(make_params(**base, fixed_order=("roth", "trad")), 6_000.0).rows[0]
        trad_first = simulate(make_params(**base, fixed_order=("trad", "roth")), 6_000.0).rows[0]
        self.assertGreater(roth_first["draw_roth"], 0.0)
        self.assertEqual(roth_first["draw_trad"], 0.0)
        self.assertEqual(roth_first["tax_federal"], 0.0)          # Roth money is untaxed
        self.assertGreater(trad_first["draw_trad"], 0.0)
        self.assertEqual(trad_first["draw_roth"], 0.0)

    def test_trad_first_beats_roth_first_when_social_security_follows(self):
        # Mechanism: drawing traditional money BEFORE benefits start keeps it from
        # stacking on top of taxable Social Security later. (Observed gap ~ $96/mo.)
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0, ss_fra_monthly1=4_000.0)
        w_trad_first = solve_max_w(make_params(**base, fixed_order=("trad", "roth")))
        w_roth_first = solve_max_w(make_params(**base, fixed_order=("roth", "trad")))
        self.assertGreater(w_trad_first, w_roth_first + 50.0)

    def test_order_nearly_irrelevant_without_social_security(self):
        # With equal returns on every account and no SS, order should barely matter.
        base = dict(trad1=1_000_000.0, roth1=1_000_000.0)
        a = solve_max_w(make_params(**base, fixed_order=("trad", "roth")))
        b = solve_max_w(make_params(**base, fixed_order=("roth", "trad")))
        self.assertLess(abs(a - b), 10.0)

    # ---- purchases ----
    def test_purchases_reduce_w(self):
        p = make_params(roth1=1_000_000.0, purchases=((2030, 40_000.0, "car"),))
        self.assertLess(solve_max_w(p), solve_max_w(p, include_purchases=False))

    def test_include_purchases_false_with_no_purchases(self):
        p = make_params(roth1=1_000_000.0)
        self.assertAlmostEqual(solve_max_w(p, include_purchases=True),
                               solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_beyond_horizon_is_ignored(self):
        p = make_params(roth1=1_000_000.0, purchases=((2100, 40_000.0, "x"),))
        self.assertAlmostEqual(solve_max_w(p), solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_before_start_is_ignored(self):
        p = make_params(roth1=1_000_000.0, purchases=((2026, 40_000.0, "x"),))
        self.assertAlmostEqual(solve_max_w(p), solve_max_w(p, include_purchases=False), places=2)

    def test_purchase_in_last_year_is_applied(self):
        base = make_params(roth1=1_000_000.0)
        last_year = base.start_year + base.n_years - 1
        with_buy = make_params(roth1=1_000_000.0, purchases=((last_year, 40_000.0, "x"),))
        self.assertLess(solve_max_w(with_buy), solve_max_w(base))

    def test_purchase_in_first_year_is_applied(self):
        with_buy = make_params(roth1=1_000_000.0, purchases=((2027, 40_000.0, "x"),))
        row = simulate(with_buy, 3_000.0).rows[0]
        self.assertAlmostEqual(row["spend_real"], 36_000.0 + 40_000.0, places=2)

    # ---- edge cases ----
    def test_zero_portfolio_returns_zero(self):
        self.assertEqual(solve_max_w(make_params()), 0.0)

    def test_zero_spending_just_compounds(self):
        p = make_params(roth1=1_000_000.0)
        res = simulate(p, 0.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.final_nominal, 1_000_000.0 * 1.05 ** p.n_years, delta=1.0)
        self.assertEqual(res.lifetime_tax, 0.0)

    def test_purchases_exceeding_portfolio_make_w_zero(self):
        # solve_max_w's other early return: even W=0 fails.
        p = make_params(roth1=1_000_000.0, purchases=((2028, 5_000_000.0, "mansion"),))
        self.assertEqual(solve_max_w(p), 0.0)

    # ---- depletion ----
    def test_depletion_year_matches_closed_form(self):
        p = make_params(roth1=1_000_000.0, strategy="fixed_order", fixed_order=("roth",))
        annual_real = 72_000.0                                  # W = 6,000/month
        rr = (1 + p.nominal_return) / (1 + p.inflation) - 1
        b, expected = 1_000_000.0, None
        for k in range(p.n_years):
            if b < annual_real:
                expected = p.start_year + k
                break
            b = (b - annual_real) * (1 + rr)
        self.assertIsNotNone(expected)                          # test setup: it does run out
        res = simulate(p, 6_000.0)
        self.assertFalse(res.success)
        self.assertEqual(res.depletion_year, expected)
        for r in res.rows:
            if r["year"] < res.depletion_year:
                self.assertAlmostEqual(r["shortfall_nominal"], 0.0)
            else:
                self.assertGreater(r["shortfall_nominal"], 0.0)

    def test_no_depletion_year_when_funded(self):
        res = simulate(make_params(roth1=1_000_000.0), 2_000.0)
        self.assertTrue(res.success)
        self.assertIsNone(res.depletion_year)


# ===========================================================================
# Social Security in the engine
# ===========================================================================
class SocialSecurityEngineTests(unittest.TestCase):
    def test_ss_closed_form_with_no_tax_no_growth(self):
        # Born 1965-01-01 -> FRA 67 -> benefits start Jan 2032, run through Dec 2059.
        p = make_params(roth1=1_200_000.0, nominal_return=0.0, inflation=0.0,
                        tax_enabled=False, ss_fra_monthly1=2_000.0,
                        strategy="fixed_order", fixed_order=("roth",))
        by_year = {r["year"]: r for r in simulate(p, 1_000.0).rows}
        # Pin the arithmetic's assumptions instead of trusting them:
        self.assertAlmostEqual(by_year[2031]["ss_gross"], 0.0)
        self.assertAlmostEqual(by_year[2032]["ss_gross"], 24_000.0, places=2)   # 12 months
        self.assertAlmostEqual(by_year[2059]["ss_gross"], 24_000.0, places=2)
        months = (2059 * 12 + 11) - (2032 * 12) + 1                             # 336
        self.assertEqual(months, 336)
        expected = (1_200_000.0 + 2_000.0 * months) / (12 * p.n_years)
        self.assertAlmostEqual(solve_max_w(p), expected, delta=0.05)

    def test_ss_raises_w(self):
        p = make_params(trad1=1_500_000.0)
        self.assertGreater(solve_max_w(replace(p, ss_fra_monthly1=3_000.0)), solve_max_w(p))

    def test_ss_taxed_when_income_is_high(self):
        p = make_params(trad1=2_000_000.0, ss_fra_monthly1=3_000.0)
        late = [r for r in simulate(p, 8_000.0).rows if r["year"] >= 2033]
        self.assertTrue(any(r["ss_taxable"] > 0 for r in late))
        self.assertTrue(all(r["ss_taxable"] <= 0.85 * r["ss_gross"] + 0.01 for r in late))

    def test_ss_benefit_multiplier(self):
        p = make_params(roth1=500_000.0, ss_fra_monthly1=2_000.0)
        full = simulate(p, 1_000.0).rows[10]["ss_gross"]
        cut = simulate(replace(p, ss_multiplier=0.77), 1_000.0).rows[10]["ss_gross"]
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(cut, 0.77 * full, places=2)

    def test_surplus_ss_goes_to_cash(self):
        p = make_params(roth1=100_000.0, ss_fra_monthly1=5_000.0)
        row = [r for r in simulate(p, 1_000.0).rows if r["year"] == 2032][0]
        self.assertGreater(row["surplus_to_cash"], 0.0)
        self.assertGreater(row["end_cash"], 0.0)


# ===========================================================================
# 59.5 rule and penalties
# ===========================================================================
class EarlyWithdrawalTests(unittest.TestCase):
    # Born 1975-01-01: turns 59.5 in Jul 2034. July is after the June 30 cutoff of
    # the mid-year convention, so access is penalty-free from 2035; the last
    # penalized year is 2034. Age 52 in 2027.
    def young(self, **kw):
        return make_params(person1_birth=date(1975, 1, 1), **kw)

    def test_locked_traditional_and_roth_both_penalized(self):
        # Small Roth runs out first (early_access_order = roth, trad), so over the
        # locked years we see Roth-only, mixed, and trad-only draws.
        p = self.young(trad1=500_000.0, roth1=60_000.0)
        res = simulate(p, 2_000.0)
        by_year = {r["year"]: r for r in res.rows}
        locked_rows = [by_year[y] for y in range(2027, 2035)]
        for r in locked_rows:
            self.assertAlmostEqual(r["penalty_early_withdrawal"],
                                   0.10 * (r["draw_trad"] + r["draw_roth"]), delta=0.02)
            self.assertEqual(r["early_withdrawal_flag"], "PENALTY")
        self.assertTrue(any(r["draw_roth"] > 0 for r in locked_rows))
        self.assertTrue(any(r["draw_trad"] > 0 for r in locked_rows))
        self.assertTrue(any(r["draw_roth"] > 0 and r["draw_trad"] > 0 for r in locked_rows))
        self.assertGreater(res.lifetime_penalty, 0.0)

    def test_lock_boundary_year(self):
        p = self.young(trad1=500_000.0)
        res = simulate(p, 2_000.0)
        by_year = {r["year"]: r for r in res.rows}
        self.assertEqual(by_year[2034]["early_withdrawal_flag"], "PENALTY")   # last locked year
        self.assertEqual(by_year[2035]["early_withdrawal_flag"], "")          # first unlocked year
        self.assertEqual(by_year[2035]["penalty_early_withdrawal"], 0.0)
        self.assertEqual(res.penalty_years, list(range(2027, 2035)))
        self.assertEqual(p.unlock_year(1), 2035)

    def test_no_penalty_when_other_money_exists(self):
        p = self.young(trad1=500_000.0, cash=400_000.0)
        res = simulate(p, 2_000.0)
        first = res.rows[0]
        self.assertEqual(res.penalty_years, [])
        self.assertAlmostEqual(first["end_trad1"], 500_000.0 * 1.05, delta=0.01)  # untouched
        self.assertGreater(first["draw_cash"], 0.0)

    def test_locked_is_per_owner(self):
        # Person 2 is older (unlocked), person 1 is young (locked). Only person 2's
        # account should be drawn in the early years, with no penalty.
        p = self.young(person2_birth=date(1960, 1, 1), trad1=500_000.0, trad2=500_000.0)
        res = simulate(p, 2_000.0)
        self.assertEqual(res.penalty_years, [])
        self.assertAlmostEqual(res.rows[0]["end_trad1"], 500_000.0 * 1.05, delta=0.01)
        self.assertLess(res.rows[0]["end_trad2"], 500_000.0 * 1.05)

    def test_penalty_raises_cost(self):
        locked = self.young(trad1=1_000_000.0)
        # Same account, but the owner is already past 59.5; horizon_age is raised
        # by the 15-year birth-year difference so the number of years is identical.
        unlocked = make_params(person1_birth=date(1960, 1, 1), trad1=1_000_000.0,
                               horizon_age=locked.horizon_age + 15)
        self.assertEqual(locked.n_years, unlocked.n_years)       # setup sanity check
        self.assertLess(solve_max_w(locked), solve_max_w(unlocked))

    def test_tax_disabled_disables_penalty(self):
        p = self.young(trad1=500_000.0, tax_enabled=False)
        self.assertEqual(simulate(p, 2_000.0).lifetime_penalty, 0.0)


# ===========================================================================
# Strategies
# ===========================================================================
class StrategyTests(unittest.TestCase):
    def test_taxable_std_then_cash_draw_amounts(self):
        # basis == value so there are no gains and taxes don't distort the draw split
        p = make_params(cash=1_000_000.0, taxable_value=1_000_000.0,
                        taxable_basis=1_000_000.0, strategy="taxable_std_then_cash")
        row = simulate(p, 8_000.0).rows[0]
        std_nominal = 32_200.0 * 1.025              # 2027 is one year of indexing past 2026
        self.assertAlmostEqual(row["draw_taxable"], std_nominal, places=2)
        self.assertAlmostEqual(row["draw_cash"], 96_000.0 - std_nominal, places=2)

    def test_taxable_std_then_cash_after_cash_runs_out(self):
        p = make_params(cash=10_000.0, taxable_value=2_000_000.0,
                        taxable_basis=2_000_000.0, strategy="taxable_std_then_cash")
        res = simulate(p, 6_000.0)      # sustainable for a $2M portfolio over 33 years
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.rows[3]["draw_cash"], 0.0, places=2)
        self.assertGreater(res.rows[3]["draw_taxable"], 32_200.0 * 1.025 ** 4)

    def test_trad_std_then_cash_fills_deduction_with_traditional(self):
        p = make_params(cash=500_000.0, trad1=1_000_000.0, strategy="trad_std_then_cash")
        row = simulate(p, 6_000.0).rows[0]
        self.assertAlmostEqual(row["draw_trad"], 32_200.0 * 1.025, places=2)
        self.assertEqual(row["tax_federal"], 0.0)   # exactly the standard deduction

    def test_strategies_do_not_touch_locked_money_first(self):
        for name in model.STRATEGIES:
            with self.subTest(strategy=name):
                p = make_params(person1_birth=date(1975, 1, 1), cash=200_000.0,
                                taxable_value=200_000.0, taxable_basis=200_000.0,
                                trad1=800_000.0, strategy=name)
                res = simulate(p, 2_000.0)
                self.assertAlmostEqual(res.rows[0]["end_trad1"], 800_000.0 * 1.05, delta=0.01)


# ===========================================================================
# Social Security claim age (v2)
# ===========================================================================
class ClaimAdjustmentTests(unittest.TestCase):
    FRA67 = date(1965, 1, 1)

    def factor(self, years, months=0, birth=None):
        return rules.ss_claim_factor(birth or self.FRA67, years * 12 + months)

    def test_known_ssa_percentages_for_fra_67(self):
        table = {62: 0.70, 63: 0.75, 64: 0.80, 65: 13 / 15, 66: 14 / 15, 67: 1.0,
                 68: 1.08, 69: 1.16, 70: 1.24}
        for age, want in table.items():
            self.assertAlmostEqual(self.factor(age), want, places=6, msg=f"age {age}")

    def test_partial_months(self):
        self.assertAlmostEqual(self.factor(66, 6), 1 - 6 * (5 / 9) / 100, places=8)
        self.assertAlmostEqual(self.factor(67, 6), 1 + 6 * (2 / 3) / 100, places=8)

    def test_default_claim_is_fra(self):
        self.assertEqual(rules.ss_claim_factor(self.FRA67, None), 1.0)

    def test_early_reduction_changes_rate_after_36_months(self):
        # 36 months early = 20%; each further month costs 5/12 of 1% (less than 5/9)
        self.assertAlmostEqual(1 - self.factor(64), 0.20, places=8)
        self.assertAlmostEqual(self.factor(64) - self.factor(63), 0.05, places=8)   # 12 mo x 5/12%
        self.assertAlmostEqual(self.factor(65) - self.factor(64), 12 * (5 / 9) / 100, places=8)

    def test_fra_66_cohort(self):
        b = date(1950, 6, 1)                                   # FRA 66y0m
        self.assertAlmostEqual(rules.ss_claim_factor(b, 62 * 12), 0.75, places=8)   # 48 months early
        self.assertAlmostEqual(rules.ss_claim_factor(b, 70 * 12), 1.32, places=8)   # 48 months delayed

    def test_fra_66_and_2_months_cohort(self):
        b = date(1955, 5, 1)                                   # FRA 66y2m = 794 months
        self.assertAlmostEqual(rules.ss_claim_factor(b, 70 * 12), 1 + 46 * (2 / 3) / 100, places=8)
        self.assertAlmostEqual(rules.ss_claim_factor(b, 62 * 12),
                               1 - 36 * (5 / 9) / 100 - 14 * (5 / 12) / 100, places=8)

    def test_window_enforced(self):
        for bad in (62 * 12 - 1, 70 * 12 + 1):
            with self.assertRaises(ValueError):
                rules.ss_claim_factor(self.FRA67, bad)
            with self.assertRaises(ValueError):
                make_params(ss_claim_months1=bad, ss_fra_monthly1=2_000.0)

    def test_claim_for_missing_person2_rejected(self):
        with self.assertRaises(ValueError):
            make_params(ss_claim_months2=67 * 12)


class ClaimTimingTests(unittest.TestCase):
    def test_first_year_is_prorated_by_month(self):
        # The cases from the household: end-of-February and mid-April birthdays, FRA 67.
        feb, apr = date(1965, 2, 28), date(1965, 4, 15)
        self.assertEqual(rules.ss_start_year_month(feb), (2032, 2))
        self.assertEqual([rules.ss_months_paid(feb, y) for y in (2031, 2032, 2033)], [0, 11, 12])
        self.assertEqual(rules.ss_start_year_month(apr), (2032, 4))
        self.assertEqual([rules.ss_months_paid(apr, y) for y in (2031, 2032, 2033)], [0, 9, 12])

    def test_start_month_follows_claim_age(self):
        apr = date(1965, 4, 15)
        self.assertEqual(rules.ss_start_year_month(apr, 70 * 12), (2035, 4))
        self.assertEqual([rules.ss_months_paid(apr, y, 70 * 12) for y in (2034, 2035, 2036)], [0, 9, 12])
        # claim at 62y6m -> Oct 2027: October, November, December = 3 months
        self.assertEqual(rules.ss_start_year_month(apr, 62 * 12 + 6), (2027, 10))
        self.assertEqual(rules.ss_months_paid(apr, 2027, 62 * 12 + 6), 3)


class ClaimEngineTests(unittest.TestCase):
    def test_claim_70_row_amounts(self):
        # inflation 0 so nominal == real. Born 1965-01-01, claim 70y0m -> Jan 2035, x1.24.
        p = make_params(roth1=500_000.0, inflation=0.0, ss_fra_monthly1=2_000.0,
                        ss_claim_months1=70 * 12)
        by_year = {r["year"]: r for r in simulate(p, 1_000.0).rows}
        self.assertAlmostEqual(by_year[2034]["ss_gross"], 0.0)
        self.assertAlmostEqual(by_year[2035]["ss_gross"], 12 * 2_000.0 * 1.24, places=2)

    def test_claim_62_row_amounts_and_partial_year(self):
        p = make_params(person1_birth=date(1965, 4, 15), roth1=500_000.0, inflation=0.0,
                        ss_fra_monthly1=2_000.0, ss_claim_months1=62 * 12)
        by_year = {r["year"]: r for r in simulate(p, 1_000.0).rows}
        # April 2027 start: 9 months of 70% benefit, then a full year
        self.assertAlmostEqual(by_year[2027]["ss_gross"], 9 * 2_000.0 * 0.70, places=2)
        self.assertAlmostEqual(by_year[2028]["ss_gross"], 12 * 2_000.0 * 0.70, places=2)

    def test_closed_form_with_claim_70(self):
        # no tax, no growth: W = (balance + total SS) / (12 N); Jan 2035..Dec 2059 = 300 months
        p = make_params(roth1=1_200_000.0, nominal_return=0.0, inflation=0.0, tax_enabled=False,
                        ss_fra_monthly1=2_000.0, ss_claim_months1=70 * 12,
                        strategy="fixed_order", fixed_order=("roth",))
        expected = (1_200_000.0 + 2_000.0 * 1.24 * 300) / (12 * p.n_years)
        self.assertAlmostEqual(solve_max_w(p), expected, delta=0.05)

    def test_claim_age_can_differ_by_person(self):
        p = make_params(person2_birth=date(1966, 1, 1), roth1=500_000.0, inflation=0.0,
                        ss_fra_monthly1=2_000.0, ss_fra_monthly2=1_000.0,
                        ss_claim_months1=62 * 12, ss_claim_months2=70 * 12)
        by_year = {r["year"]: r for r in simulate(p, 1_000.0).rows}
        self.assertAlmostEqual(by_year[2027]["ss_gross"], 12 * 2_000.0 * 0.70, places=2)   # p1 only
        self.assertAlmostEqual(by_year[2036]["ss_gross"],
                               12 * 2_000.0 * 0.70 + 12 * 1_000.0 * 1.24, places=2)         # both

    def test_fixed_death_at_95_favors_delaying(self):
        # Known caveat (DESIGN.md): with no death dates, W rises with claim age.
        # Observed gaps ~ $300 (62->67) and ~ $170 (67->70) per month in this scenario.
        base = dict(trad1=800_000.0, roth1=200_000.0, ss_fra_monthly1=3_000.0)
        w = {age: solve_max_w(make_params(**base, ss_claim_months1=age * 12)) for age in (62, 67, 70)}
        self.assertGreater(w[67], w[62] + 200.0)
        self.assertGreater(w[70], w[67] + 100.0)

    def test_delay_advantage_shrinks_when_returns_rise(self):
        base = dict(trad1=800_000.0, roth1=200_000.0, ss_fra_monthly1=3_000.0)
        gap = lambda r: (solve_max_w(make_params(**base, nominal_return=r, ss_claim_months1=70 * 12))
                         - solve_max_w(make_params(**base, nominal_return=r, ss_claim_months1=62 * 12)))
        self.assertGreater(gap(0.05), gap(0.08))

    def test_row_price_index_deflates_nominal_to_real(self):
        p = make_params(roth1=500_000.0)
        for r in simulate(p, 3_000.0).rows:
            self.assertAlmostEqual(r["spend_nominal"] / r["price_index"], r["spend_real"], delta=0.02)


class ConfigLoadingTests(unittest.TestCase):
    MINIMAL = """
[household]
person1_birth = 1965-06-15
person2_birth = 1966-09-02
filing_status = "mfj"
[model]
start_year = 2027
nominal_return = 0.05
inflation = 0.025
[balances]
roth_person1 = 100000
[social_security]
fra_monthly_person1 = 2000
fra_monthly_person2 = 1000
{claims}
[spending]
target_monthly_real = 1000
"""

    def load(self, claims=""):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(self.MINIMAL.format(claims=claims))
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    def test_claim_ages_parsed_to_months(self):
        p = self.load("claim_age_person1 = [70, 0]\nclaim_age_person2 = [66, 6]")
        self.assertEqual((p.ss_claim_months1, p.ss_claim_months2), (840, 798))

    def test_claim_ages_default_to_fra(self):
        p = self.load()
        self.assertIsNone(p.ss_claim_months1)
        self.assertIsNone(p.ss_claim_months2)

    def test_malformed_claim_ages_rejected(self):
        for bad in ("claim_age_person1 = [67]", "claim_age_person1 = [67, 12]",
                    "claim_age_person1 = 67", "claim_age_person1 = [67.5, 0]",
                    "claim_age_person1 = [61, 11]", "claim_age_person1 = [70, 1]"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.load(bad)

    def test_shipped_template_loads_and_runs(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.template.toml")
        p = model.load_params(path)
        self.assertTrue(simulate(p, 5_000.0).rows)


# ===========================================================================
# Deaths, filing status, survivor benefits (v3)
# ===========================================================================
B65 = date(1965, 1, 1)          # FRA 67 -> benefits from Jan 2032 when claiming at FRA


def idx(year, month):           # month index, as used inside rules.py
    return rules.month_index(year, month)


class SurvivorRuleTests(unittest.TestCase):
    """Pure-function checks of the survivor-benefit rules (rules.py)."""

    def test_basis_when_deceased_had_claimed_at_fra(self):
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, None, idx(2045, 7)), 1.0)

    def test_basis_when_deceased_claimed_at_70(self):
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 70 * 12, idx(2045, 7)), 1.24)

    def test_widow_limit_floor_when_deceased_claimed_early(self):
        # claimed at 62 (70% of PIA), then died: survivor gets at least 82.5% of PIA
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 62 * 12, idx(2045, 7)), 0.825)
        # claimed at 66 (93.3%): above the floor, so their own reduced benefit stands
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 66 * 12, idx(2045, 7)), 14 / 15)

    def test_basis_when_deceased_died_before_claiming_and_before_fra(self):
        # planned to claim at 70, died at 64 (2029): survivor basis is 100% of PIA, not 124%
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 70 * 12, idx(2029, 6)), 1.0)

    def test_basis_when_deceased_died_after_fra_before_claiming(self):
        # planned 70, died at 68y0m (Jan 2033): earned 12 months of delayed credits = 8%
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 70 * 12, idx(2033, 1)), 1.08)
        # died at 69y6m (Jul 2034), still unclaimed: 30 months of credits = 20%
        self.assertAlmostEqual(rules.survivor_basis_factor(B65, 70 * 12, idx(2034, 7)), 1.20)

    def test_survivor_start_gated_by_age_60_with_linear_reduction(self):
        # survivor born 1980-01 is 50 at the 2030 death -> starts at 60 (Jan 2040), 28.5% off
        start, redn = rules.survivor_start_and_reduction(date(1980, 1, 1), idx(2030, 6))
        self.assertEqual(start, idx(2040, 1))
        self.assertAlmostEqual(redn, 1 - 0.285)
        # survivor 63 (FRA 67): 48 of 84 months early
        start, redn = rules.survivor_start_and_reduction(B65, idx(2028, 1))
        self.assertEqual(start, idx(2028, 1))
        self.assertAlmostEqual(redn, 1 - 0.285 * 48 / 84)
        # survivor past FRA: no reduction, starts at death
        start, redn = rules.survivor_start_and_reduction(B65, idx(2045, 7))
        self.assertEqual((start, redn), (idx(2045, 7), 1.0))

    def household(self, persons, first=2043, last=2047):
        # these tests are about SURVIVOR rules: the spousal top-up is switched off (it has its own tests)
        return rules.ss_real_by_year(persons, first, last, spousal=False)

    def test_higher_earner_dies_survivor_keeps_larger_check(self):
        r = self.household([(B65, 3000, None, date(2045, 7, 15)), (B65, 1000, None, date(2060, 1, 1))])
        self.assertAlmostEqual(r[2044], 12 * 4000)
        self.assertAlmostEqual(r[2045], 6 * 4000 + 6 * 3000)     # death month onward = one check
        self.assertAlmostEqual(r[2046], 12 * 3000)

    def test_lower_earner_dies_survivor_keeps_own_larger_check(self):
        r = self.household([(B65, 3000, None, date(2060, 1, 1)), (B65, 1000, None, date(2045, 7, 15))])
        self.assertAlmostEqual(r[2045], 6 * 4000 + 6 * 3000)
        self.assertAlmostEqual(r[2046], 12 * 3000)

    def test_zero_own_benefit_spouse_gets_survivor_benefit(self):
        r = self.household([(B65, 3000, None, date(2045, 7, 15)), (B65, 0, None, date(2060, 1, 1))])
        self.assertAlmostEqual(r[2044], 12 * 3000)
        self.assertAlmostEqual(r[2046], 12 * 3000)               # survivor receives p1's benefit

    def test_benefits_stop_after_second_death(self):
        r = self.household([(B65, 3000, None, date(2045, 7, 15)), (B65, 1000, None, date(2046, 3, 1))],
                           2045, 2047)
        self.assertAlmostEqual(r[2046], 2 * 3000)                # Jan, Feb only; none from the March death month
        self.assertAlmostEqual(r[2047], 0.0)

    def test_same_month_deaths_leave_nothing(self):
        r = self.household([(B65, 3000, None, date(2045, 7, 15)), (B65, 1000, None, date(2045, 7, 20))],
                           2045, 2046)
        self.assertAlmostEqual(r[2045], 6 * 4000)
        self.assertAlmostEqual(r[2046], 0.0)

    def test_survivor_switches_to_larger_own_benefit_when_it_starts(self):
        # p2 (PIA 1000, claims at FRA) dies Aug 2033. p1 (PIA 3000) claims at 70 (Jan 2035, x1.24).
        # Between: survivor benefit 1000. From Jan 2035: own 3720 beats it.
        r = self.household([(B65, 3000, 70 * 12, date(2060, 1, 1)), (B65, 1000, None, date(2033, 8, 1))],
                           2033, 2036)
        self.assertAlmostEqual(r[2033], 12 * 1000)       # Jan-Jul: p2's own 1000; Aug-Dec: survivor benefit 1000
        self.assertAlmostEqual(r[2034], 12 * 1000)
        self.assertAlmostEqual(r[2035], 12 * 3000 * 1.24)

    def test_early_death_before_claiming_uses_pia_for_survivor(self):
        # p1 plans to claim at 70 but dies at 64 (Jun 2029). Survivor p2 (also born 1965) is 64:
        # benefit = 100% of PIA, reduced for taking it 36 months before FRA 67 (84-month scale).
        r = self.household([(B65, 3000, 70 * 12, date(2029, 6, 1)), (B65, 1000, None, date(2060, 1, 1))],
                           2029, 2031)
        start_age = idx(2029, 6) - idx(1965, 1)                      # survivor's age in months at death
        redn = 1 - 0.285 * (804 - start_age) / 84
        self.assertAlmostEqual(r[2030], 12 * max(3000 * redn, 1000))

    def test_single_person_has_no_survivor_logic(self):
        r = self.household([(B65, 3000, None, date(2045, 7, 15))], 2044, 2046)
        self.assertAlmostEqual(r[2044], 12 * 3000)
        self.assertAlmostEqual(r[2045], 6 * 3000)                # no payment from the death month
        self.assertAlmostEqual(r[2046], 0.0)


class DeathAndFilingTests(unittest.TestCase):
    def couple(self, **kw):
        base = dict(person2_birth=date(1966, 1, 1), person1_death=date(2045, 7, 15),
                    person2_death=date(2060, 1, 1), roth1=1_500_000.0, inflation=0.0,
                    ss_fra_monthly1=3_000.0, ss_fra_monthly2=1_000.0, ss_spousal=False)
        base.update(kw)
        return make_params(**base)

    def test_filing_status_switches_the_year_after_first_death(self):
        p = self.couple()
        self.assertEqual([p.filing_status_for(y) for y in (2044, 2045, 2046)], ["mfj", "mfj", "single"])

    def test_configured_single_stays_single(self):
        p = self.couple(filing_status="single")
        self.assertEqual({p.filing_status_for(y) for y in (2027, 2045, 2050)}, {"single"})

    def test_rows_show_filing_status_and_living_count(self):
        by_year = {r["year"]: r for r in simulate(self.couple(), 2_000.0).rows}
        self.assertEqual((by_year[2045]["filing_status"], by_year[2045]["n_living"]), ("mfj", 2))
        self.assertEqual((by_year[2046]["filing_status"], by_year[2046]["n_living"]), ("single", 1))

    def test_household_social_security_drops_to_single_largest_benefit(self):
        by_year = {r["year"]: r for r in simulate(self.couple(), 2_000.0).rows}
        # p1 born 1965 claims Jan 2032 (3,000), p2 born 1966 claims Jan 2033 (1,000); inflation 0
        self.assertAlmostEqual(by_year[2044]["ss_gross"], 12 * 4_000.0, places=2)
        self.assertAlmostEqual(by_year[2045]["ss_gross"], 6 * 4_000.0 + 6 * 3_000.0, places=2)
        self.assertAlmostEqual(by_year[2046]["ss_gross"], 12 * 3_000.0, places=2)

    def test_single_standard_deduction_cap_after_first_death(self):
        # trad_std_then_cash fills one standard deduction from traditional money;
        # after the first death that deduction is the single-filer amount.
        # born 1980/1981: RMDs (age 75) and Social Security are decades away, so only the cap matters
        p = self.couple(person1_birth=date(1980, 1, 1), person2_birth=date(1981, 1, 1),
                        roth1=0.0, trad1=1_500_000.0, cash=3_000_000.0, nominal_return=0.0,
                        strategy="trad_std_then_cash")
        by_year = {r["year"]: r for r in simulate(p, 8_000.0).rows}     # spending well above SS
        self.assertAlmostEqual(by_year[2045]["draw_trad"], 32_200.0, places=2)      # still joint
        self.assertAlmostEqual(by_year[2046]["draw_trad"], 16_100.0, places=2)      # single

    def test_status_reaches_the_tax_calculation(self):
        base = dict(trad1=2_000_000.0, inflation=0.0, nominal_return=0.0)
        mfj = simulate(make_params(**base, filing_status="mfj"), 6_000.0).rows[0]
        single = simulate(make_params(**base, filing_status="single"), 6_000.0).rows[0]
        self.assertGreater(single["tax_federal"], mfj["tax_federal"])

    def test_taxes_rise_after_the_switch_to_single(self):
        # No Social Security, same spending: the ONLY difference is when the first death occurs.
        base = dict(person2_birth=date(1966, 1, 1), person2_death=date(2060, 1, 1),
                    trad1=3_000_000.0, inflation=0.0, nominal_return=0.0)
        early = {r["year"]: r for r in simulate(make_params(**base, person1_death=date(2045, 7, 15)), 8_000.0).rows}
        late = {r["year"]: r for r in simulate(make_params(**base, person1_death=date(2059, 7, 15)), 8_000.0).rows}
        self.assertAlmostEqual(early[2045]["tax_federal"], late[2045]["tax_federal"], places=2)   # still joint
        self.assertEqual(early[2046]["filing_status"], "single")
        self.assertEqual(late[2046]["filing_status"], "mfj")
        self.assertGreater(early[2046]["tax_federal"], late[2046]["tax_federal"])

    def test_deceased_accounts_unlock_for_the_survivor_the_year_after_death(self):
        # person 2 (born 1975, locked until 2035) dies in 2030; person 1 (born 1960) survives
        p = make_params(person1_birth=date(1960, 1, 1), person2_birth=date(1975, 1, 1),
                        person2_death=date(2030, 6, 1), trad1=100_000.0, trad2=100_000.0)
        self.assertIn("trad2", p.locked_buckets(2029))
        self.assertIn("trad2", p.locked_buckets(2030))            # death year: owner's own status
        self.assertNotIn("trad2", p.locked_buckets(2031))
        self.assertNotIn("roth2", p.locked_buckets(2031))

    def test_survivor_can_spend_deceased_locked_account_without_penalty(self):
        p = make_params(person1_birth=date(1960, 1, 1), person2_birth=date(1975, 1, 1),
                        person2_death=date(2030, 6, 1), trad2=800_000.0, cash=0.0,
                        person1_death=date(2060, 1, 1), nominal_return=0.0, inflation=0.0)
        res = simulate(p, 2_000.0)
        # person 2's locked account is the only money: penalty before their death, none after
        by_year = {r["year"]: r for r in res.rows}
        self.assertGreater(by_year[2029]["penalty_early_withdrawal"], 0.0)
        self.assertEqual(by_year[2032]["penalty_early_withdrawal"], 0.0)
        self.assertGreater(by_year[2032]["draw_trad"], 0.0)

    def test_horizon_ends_at_last_death(self):
        p = self.couple()
        res = simulate(p, 2_000.0)
        self.assertEqual(res.rows[-1]["year"], 2059)               # last death 2060 -> last year 2059
        self.assertEqual(len(res.rows), p.n_years)

    def test_default_deaths_match_old_horizon_behavior(self):
        # no death dates: both die at 95; the first death is the older person's 95th birthday
        p = make_params(person2_birth=date(1966, 1, 1), roth1=1_000_000.0)
        self.assertEqual(p.n_years, 1966 + 95 - 2027)
        by_year = {r["year"]: r for r in simulate(p, 2_000.0).rows}
        self.assertEqual(by_year[2060]["filing_status"], "mfj")      # first-death year (p1 b.1965) stays joint


class DeathConfigTests(unittest.TestCase):
    TEMPLATE = ConfigLoadingTests.MINIMAL

    def load(self, household_extra="", model_extra=""):
        text = self.TEMPLATE.format(claims="")
        text = text.replace('filing_status = "mfj"', 'filing_status = "mfj"\n' + household_extra)
        text = text.replace("inflation = 0.025", "inflation = 0.025\n" + model_extra)
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    def test_death_dates_load(self):
        p = self.load("person1_death = 2057-06-15\nperson2_death = 2062-09-02")
        self.assertEqual((p.person1_death, p.person2_death), (date(2057, 6, 15), date(2062, 9, 2)))
        self.assertEqual((p.first_death_year, p.last_death_year), (2057, 2062))

    def test_death_dates_default_to_none(self):
        p = self.load()
        self.assertIsNone(p.person1_death)
        self.assertEqual(p.death1, date(2060, 6, 15))

    def test_string_death_date_rejected(self):
        with self.assertRaises(ValueError):
            self.load('person1_death = "2057-06-15"')

    def test_legacy_horizon_person_handled(self):
        self.assertEqual(self.load(model_extra='horizon_person = "younger"').n_years,
                         self.load().n_years)                      # old default is accepted silently
        with self.assertRaises(ValueError):
            self.load(model_extra='horizon_person = "person1"')   # semantics changed: must use death dates


# ===========================================================================
# Required minimum distributions (v4)
# ===========================================================================
def ult(age):
    return rules.UNIFORM_LIFETIME[min(age, 120)]


class RMDRuleTests(unittest.TestCase):
    def test_rmd_age_by_birth_year(self):
        self.assertEqual([rules.rmd_age(y) for y in (1950, 1951, 1959, 1960, 1972)],
                         [72, 73, 73, 75, 75])

    def test_table_checkpoints_and_shape(self):
        table = rules.UNIFORM_LIFETIME
        self.assertEqual(len(table), 49)                              # ages 72..120
        for age, want in {72: 27.4, 73: 26.5, 75: 24.6, 80: 20.2, 85: 16.0, 90: 12.2,
                          95: 8.9, 100: 6.4, 110: 3.5, 120: 2.0}.items():
            self.assertEqual(table[age], want, msg=f"age {age}")
        divisors = [table[a] for a in range(72, 121)]
        self.assertEqual(divisors, sorted(divisors, reverse=True))     # never increases with age
        self.assertEqual(rules.rmd_divisor(125), 2.0)                  # older than the table
        with self.assertRaises(ValueError):
            rules.rmd_divisor(71)

    def test_irs_published_examples(self):
        # Pub 590-B: $100,000 at 75 -> $4,065; common worked examples for $500,000
        self.assertAlmostEqual(rules.rmd_amount(100_000, 75, 75), 4_065.04, places=2)
        for age, want in {73: 18_868, 75: 20_325, 78: 22_727, 79: 23_697, 90: 40_984, 95: 56_180}.items():
            self.assertEqual(round(rules.rmd_amount(500_000, age, 73)), want, msg=f"age {age}")

    def test_no_rmd_before_start_age_or_for_empty_account(self):
        self.assertEqual(rules.rmd_amount(1_000_000, 74, 75), 0.0)
        self.assertEqual(rules.rmd_amount(0.0, 80, 75), 0.0)


class RMDEngineTests(unittest.TestCase):
    """Zero growth, zero inflation and no tax unless stated, so the amounts are exact."""

    def base(self, **kw):
        d = dict(person1_birth=date(1960, 6, 1), trad1=1_000_000.0, cash=10_000_000.0,
                 nominal_return=0.0, inflation=0.0, tax_enabled=False)
        d.update(kw)
        return make_params(**d)

    def test_rmd_starts_at_the_start_age_and_chains_on_prior_year_end_balance(self):
        by_year = {r["year"]: r for r in simulate(self.base(), 2_000.0).rows}    # born 1960 -> age 75 in 2035
        self.assertEqual(by_year[2034]["rmd_required"], 0.0)
        b = 1_000_000.0
        for age in (75, 76, 77):
            rmd = b / ult(age)
            row = by_year[1960 + age]
            self.assertAlmostEqual(row["rmd_required"], rmd, delta=0.01)
            self.assertAlmostEqual(row["draw_trad"], rmd, delta=0.01)
            b -= rmd
            self.assertAlmostEqual(row["end_trad1"], b, delta=0.01)

    def test_rmd_larger_than_spending_is_saved_to_cash(self):
        rows = {r["year"]: r for r in simulate(self.base(), 1_000.0).rows}       # need 12,000
        rmd = 1_000_000.0 / ult(75)
        self.assertAlmostEqual(rows[2035]["draw_cash"], 0.0)
        self.assertAlmostEqual(rows[2035]["surplus_to_cash"], rmd - 12_000.0, delta=0.01)
        self.assertAlmostEqual(rows[2035]["end_cash"] - rows[2034]["end_cash"], rmd - 12_000.0, delta=0.01)

    def test_spending_beyond_the_rmd_is_covered_by_the_plan(self):
        row = {r["year"]: r for r in simulate(self.base(), 6_000.0).rows}[2035]  # need 72,000
        rmd = 1_000_000.0 / ult(75)
        self.assertAlmostEqual(row["draw_trad"], rmd, delta=0.01)
        self.assertAlmostEqual(row["draw_cash"], 72_000.0 - rmd, delta=0.01)
        self.assertAlmostEqual(row["surplus_to_cash"], 0.0)

    def test_rmd_is_ordinary_income(self):
        p = self.base(person1_birth=date(1950, 1, 1), trad1=3_000_000.0, tax_enabled=True)   # age 77 in 2027
        row = simulate(p, 2_000.0).rows[0]
        rmd = 3_000_000.0 / ult(77)
        self.assertAlmostEqual(row["rmd_required"], rmd, delta=0.01)
        self.assertAlmostEqual(row["tax_federal"], taxes.federal_tax(rmd, 0.0, "mfj"), delta=0.01)
        self.assertGreaterEqual(row["agi"], rmd - 0.01)

    def test_roth_has_no_rmd(self):
        p = self.base(person1_birth=date(1950, 1, 1), trad1=0.0, roth1=2_000_000.0)
        self.assertTrue(all(r["rmd_required"] == 0.0 for r in simulate(p, 2_000.0).rows))

    def test_no_rmd_before_start_age_default_born_1965(self):
        rows = {r["year"]: r for r in simulate(self.base(person1_birth=date(1965, 1, 1)), 2_000.0).rows}
        self.assertTrue(all(rows[y]["rmd_required"] == 0.0 for y in range(2027, 2040)))
        self.assertGreater(rows[2040]["rmd_required"], 0.0)                        # age 75

    def test_each_persons_rmd_uses_only_their_own_account(self):
        # person 1 (b.1950) is past the start age; person 2 (b.1990) is not: spouses cannot combine
        p = self.base(person1_birth=date(1950, 1, 1), person2_birth=date(1990, 1, 1),
                      trad1=500_000.0, trad2=2_000_000.0)
        row = simulate(p, 2_000.0).rows[0]
        self.assertAlmostEqual(row["rmd_required"], 500_000.0 / ult(77), delta=0.01)
        self.assertAlmostEqual(row["end_trad2"], 2_000_000.0)

    def test_survivor_rmd_uses_combined_balance_and_survivor_age(self):
        p = self.base(person1_birth=date(1950, 1, 1), person2_birth=date(1952, 1, 1),
                      person1_death=date(2030, 6, 1), person2_death=date(2050, 1, 1),
                      trad1=600_000.0, trad2=400_000.0, cash=20_000_000.0)
        rows = {r["year"]: r for r in simulate(p, 2_000.0).rows}
        b1, b2 = 600_000.0, 400_000.0
        for year in range(2027, 2034):
            a1, a2 = year - 1950, year - 1952
            if year <= 2030:                        # both alive at the start of the year (2030 = death year)
                r1, r2 = b1 / ult(a1), b2 / ult(a2)
                b1, b2, expected = b1 - r1, b2 - r2, r1 + r2
            else:                                   # survivor: combined balance, survivor's age
                expected = (b1 + b2) / ult(a2)
                from_own = min(expected, b2)
                b2, b1 = b2 - from_own, b1 - (expected - from_own)
            self.assertAlmostEqual(rows[year]["rmd_required"], expected, delta=0.01, msg=str(year))

    def test_rmd_counts_against_the_standard_deduction_cap(self):
        # trad_std_then_cash fills one standard deduction from traditional money, RMD included
        big = simulate(self.base(person1_birth=date(1950, 1, 1), trad1=3_000_000.0,
                                 strategy="trad_std_then_cash"), 8_000.0).rows[0]
        self.assertAlmostEqual(big["draw_trad"], 3_000_000.0 / ult(77), delta=0.01)   # RMD > 32,200: no extra
        small = simulate(self.base(person1_birth=date(1950, 1, 1), trad1=500_000.0,
                                   strategy="trad_std_then_cash"), 8_000.0).rows[0]
        self.assertGreater(500_000.0 / ult(77), 0.0)
        self.assertAlmostEqual(small["draw_trad"], 32_200.0, delta=0.01)               # RMD topped up to the cap

    def test_rmd_forced_helper_never_exceeds_balances(self):
        p = self.base(person1_birth=date(1950, 1, 1), trad1=1_000.0)
        state = {"cash": 0.0, "taxable": 0.0, "trad1": 1_000.0, "trad2": 0.0, "roth1": 5e6, "roth2": 0.0}
        forced, total = model._rmd_forced(p, 2027, state)
        self.assertAlmostEqual(total, 1_000.0 / ult(77))
        self.assertEqual(forced["roth1"], 0.0)
        self.assertLessEqual(forced["trad1"], state["trad1"])

    def test_rmds_not_taken_in_a_year_after_the_owner_dies(self):
        p = self.base(person1_birth=date(1950, 1, 1), person2_birth=date(1952, 1, 1),
                      person1_death=date(2030, 6, 1), person2_death=date(2050, 1, 1),
                      trad1=600_000.0, trad2=0.0)
        rows = {r["year"]: r for r in simulate(p, 2_000.0).rows}
        self.assertGreater(rows[2030]["rmd_required"], 0.0)       # death year: deceased's RMD still due
        # after death the survivor (trad2 empty) rolls trad1 in: RMD uses the survivor's age
        self.assertAlmostEqual(rows[2031]["rmd_required"], rows[2030]["end_trad1"] / ult(2031 - 1952), delta=0.01)


# ===========================================================================
# State income tax (v5)
# ===========================================================================
class StateTaxFunctionTests(unittest.TestCase):
    def test_indiana_rate_schedule(self):
        rates = [statetax.flat_rate("IN", y) for y in (2023, 2024, 2025, 2026, 2027, 2040)]
        self.assertEqual(rates, [0.0305, 0.0305, 0.0300, 0.0295, 0.0290, 0.0290])  # last rate continues

    def test_indiana_hand_calculation(self):
        # AGI 100k incl. 20k taxable Social Security (exempt in Indiana), two people under 65:
        # base = 100,000 - 20,000 - 2 x 1,000 = 78,000 at 2.95%
        self.assertAlmostEqual(statetax.income_tax("IN", 2026, 100_000, 20_000, [60, 61]), 78_000 * 0.0295)

    def test_senior_exemption(self):
        self.assertAlmostEqual(statetax.income_tax("IN", 2026, 100_000, 0, [65, 66]),
                               (100_000 - 2 * 1_000 - 2 * 1_000) * 0.0295)
        self.assertAlmostEqual(statetax.income_tax("IN", 2026, 100_000, 0, [64, 66]),
                               (100_000 - 2 * 1_000 - 1 * 1_000) * 0.0295)       # only one is 65+

    def test_social_security_only_income_is_untaxed(self):
        self.assertEqual(statetax.income_tax("IN", 2030, 40_000, 40_000, [70]), 0.0)

    def test_base_never_negative(self):
        self.assertEqual(statetax.income_tax("IN", 2030, 1_500, 0, [40, 40]), 0.0)

    def test_no_income_tax_states(self):
        for st in ("TX", "FL"):
            self.assertEqual(statetax.income_tax(st, 2030, 500_000, 0, [60, 60], local_rate=0.0), 0.0)

    def test_local_rate_is_added_to_the_state_rate(self):
        self.assertAlmostEqual(statetax.income_tax("IN", 2026, 52_000, 0, [50, 50], 0.017),
                               50_000 * (0.0295 + 0.017))

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            statetax.check_state("ZZ")


class StateParamsTests(unittest.TestCase):
    def test_move_needs_both_date_and_state(self):
        with self.assertRaises(ValueError):
            make_params(state_origin="IN", move_date=date(2035, 6, 1))
        with self.assertRaises(ValueError):
            make_params(state_origin="IN", state_dest="TX")

    def test_move_needs_a_starting_state(self):
        with self.assertRaises(ValueError):
            make_params(state_dest="TX", move_date=date(2035, 6, 1))

    def test_move_before_start_rejected(self):
        with self.assertRaises(ValueError):
            make_params(state_origin="IN", state_dest="TX", move_date=date(2026, 6, 1))

    def test_unknown_state_and_negative_local_rate_rejected(self):
        with self.assertRaises(ValueError):
            make_params(state_origin="ZZ")
        with self.assertRaises(ValueError):
            make_params(state_origin="IN", local_tax_rate=-0.01)

    def test_origin_months(self):
        p = make_params(state_origin="IN", state_dest="TX", move_date=date(2035, 6, 15))
        self.assertEqual([p.origin_months(y) for y in (2034, 2035, 2036)], [12, 5, 0])   # Jan-May in IN
        jan = make_params(state_origin="IN", state_dest="TX", move_date=date(2035, 1, 1))
        dec = make_params(state_origin="IN", state_dest="TX", move_date=date(2035, 12, 31))
        self.assertEqual((jan.origin_months(2035), dec.origin_months(2035)), (0, 11))
        self.assertEqual(make_params(state_origin="IN").origin_months(2050), 12)          # no move


class StateEngineTests(unittest.TestCase):
    """One person born 1960 (67 in 2027, so no 59.5 lock), traditional money only, no growth."""

    def base(self, **kw):
        d = dict(person1_birth=date(1960, 1, 1), trad1=3_000_000.0, nominal_return=0.0, inflation=0.0)
        d.update(kw)
        return make_params(**d)

    def test_no_state_means_no_state_tax(self):
        res = simulate(self.base(), 5_000.0)
        self.assertTrue(all(r["tax_state"] == 0.0 and r["state_residence"] == "" for r in res.rows))
        self.assertEqual(res.lifetime_state_tax, 0.0)

    def test_indiana_tax_matches_hand_calculation(self):
        row = simulate(self.base(state_origin="IN"), 5_000.0).rows[0]      # 2027: 2.9%
        exemptions = 1_000.0 + 1_000.0                                      # personal + age 67
        self.assertGreater(row["agi"], exemptions)
        self.assertAlmostEqual(row["tax_state"], (row["agi"] - exemptions) * 0.029, delta=0.02)
        self.assertEqual(row["state_residence"], "IN")

    def test_social_security_is_not_state_taxed(self):
        p = self.base(state_origin="IN", ss_fra_monthly1=3_000.0)
        row = [r for r in simulate(p, 7_000.0).rows if r["year"] == 2040][0]
        self.assertGreater(row["ss_taxable"], 0.0)
        base = row["agi"] - row["ss_taxable"] - 2_000.0
        self.assertAlmostEqual(row["tax_state"], base * 0.029, delta=0.02)

    def test_state_tax_is_part_of_the_after_tax_cost(self):
        for r in simulate(self.base(state_origin="IN", local_tax_rate=0.017), 5_000.0).rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"] + r["shortfall_nominal"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)

    def test_state_tax_lowers_w_and_local_tax_lowers_it_more(self):
        none = solve_max_w(self.base())
        indiana = solve_max_w(self.base(state_origin="IN"))
        county = solve_max_w(self.base(state_origin="IN", local_tax_rate=0.017))
        self.assertLess(indiana, none - 20.0)
        self.assertLess(county, indiana - 20.0)

    def test_texas_and_florida_cost_nothing(self):
        none = solve_max_w(self.base())
        for st in ("TX", "FL"):
            self.assertAlmostEqual(solve_max_w(self.base(state_origin=st)), none, places=2)

    def test_moving_to_a_no_tax_state_raises_w(self):
        stay = solve_max_w(self.base(state_origin="IN", local_tax_rate=0.017))
        move = solve_max_w(self.base(state_origin="IN", local_tax_rate=0.017,
                                     state_dest="TX", move_date=date(2035, 1, 1)))
        self.assertGreater(move, stay + 10.0)

    def test_move_year_is_prorated_and_tax_stops_afterwards(self):
        p = self.base(state_origin="IN", state_dest="TX", move_date=date(2035, 7, 1))
        by_year = {r["year"]: r for r in simulate(p, 5_000.0).rows}
        self.assertEqual((by_year[2034]["state_residence"], by_year[2034]["state_origin_months"]), ("IN", 12))
        self.assertEqual((by_year[2035]["state_residence"], by_year[2035]["state_origin_months"]), ("TX", 6))
        self.assertEqual((by_year[2036]["state_residence"], by_year[2036]["state_origin_months"]), ("TX", 0))
        full_year = statetax.income_tax("IN", 2035, by_year[2035]["agi"], by_year[2035]["ss_taxable"], [75])
        self.assertAlmostEqual(by_year[2035]["tax_state"], full_year * 6 / 12, delta=0.02)
        self.assertEqual(by_year[2036]["tax_state"], 0.0)
        self.assertGreater(by_year[2034]["tax_state"], 0.0)

    def test_local_rate_applies_only_while_living_in_the_starting_state(self):
        p = self.base(state_origin="IN", state_dest="TX", move_date=date(2035, 1, 1), local_tax_rate=0.02)
        by_year = {r["year"]: r for r in simulate(p, 5_000.0).rows}
        row = by_year[2034]
        self.assertAlmostEqual(row["tax_state"], (row["agi"] - 2_000.0) * (0.029 + 0.02), delta=0.02)
        self.assertEqual(by_year[2035]["tax_state"], 0.0)

    def test_marginal_rate_includes_state(self):
        none = simulate(self.base(), 5_000.0).rows[0]["marginal_ordinary_rate"]
        indiana = simulate(self.base(state_origin="IN"), 5_000.0).rows[0]["marginal_ordinary_rate"]
        self.assertAlmostEqual(indiana - none, 0.029, delta=0.0015)

    def test_tax_disabled_disables_state_tax(self):
        res = simulate(self.base(state_origin="IN", tax_enabled=False), 5_000.0)
        self.assertEqual(res.lifetime_state_tax, 0.0)

    def test_exemptions_follow_the_people_on_the_return(self):
        p = make_params(person1_birth=date(1990, 1, 1), person2_birth=date(1990, 1, 1),   # 55 in 2045: no senior exemption
                        person1_death=date(2045, 6, 1), person2_death=date(2060, 1, 1), state_origin="IN")
        rate = lambda y: statetax.flat_rate("IN", y)
        self.assertAlmostEqual(model._state_tax_fn(p, 2045)(50_000, 0), (50_000 - 2_000) * rate(2045))  # both alive at the start
        self.assertAlmostEqual(model._state_tax_fn(p, 2046)(50_000, 0), (50_000 - 1_000) * rate(2046))  # survivor only
        senior = make_params(person1_birth=date(1960, 1, 1), state_origin="IN")
        self.assertAlmostEqual(model._state_tax_fn(senior, 2030)(50_000, 0), (50_000 - 2_000) * rate(2030))  # age 70

    def test_no_state_gives_no_tax_function(self):
        self.assertIsNone(model._state_tax_fn(make_params(), 2030))


class StateConfigTests(unittest.TestCase):
    def load(self, state_block=""):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(ConfigLoadingTests.MINIMAL.format(claims=state_block))
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    def test_no_state_block_means_no_state_tax(self):
        p = self.load()
        self.assertEqual((p.state_origin, p.state_dest, p.move_date), (None, None, None))

    def test_state_without_move(self):
        p = self.load('[state]\nresidence = "in"\nlocal_tax_rate = 0.017')
        self.assertEqual((p.state_origin, p.move_date, p.local_tax_rate), ("IN", None, 0.017))

    def test_state_with_move(self):
        p = self.load('[state]\nresidence = "IN"\nmove_date = 2035-06-01\nmove_state = "tx"')
        self.assertEqual((p.state_origin, p.state_dest, p.move_date), ("IN", "TX", date(2035, 6, 1)))

    def test_move_date_without_move_state_rejected(self):
        with self.assertRaises(ValueError):
            self.load('[state]\nresidence = "IN"\nmove_date = 2035-06-01')

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            self.load('[state]\nresidence = "ZZ"')

    def test_string_move_date_rejected(self):
        with self.assertRaises(ValueError):
            self.load('[state]\nresidence = "IN"\nmove_date = "2035-06-01"\nmove_state = "TX"')


# ===========================================================================
# Data dictionary (v5)
# ===========================================================================
class DataDictionaryTests(unittest.TestCase):
    def rows(self):
        p = make_params(person2_birth=date(1968, 1, 1), trad1=500_000.0, roth1=100_000.0, cash=50_000.0,
                        state_origin="IN", ss_fra_monthly1=2_000.0)
        return p, simulate(p, 3_000.0).rows

    def test_documented_columns_match_the_csv_columns_exactly_and_in_order(self):
        _, rows = self.rows()
        self.assertEqual([name for name, _, _ in columns.COLUMN_DOCS], list(rows[0].keys()))

    def test_every_entry_has_a_description_and_a_calculation(self):
        for name, what, how in columns.COLUMN_DOCS:
            self.assertTrue(what.strip() and how.strip(), msg=name)
            self.assertGreater(len(how), 20, msg=name)

    def test_rendered_text_names_every_column_and_this_runs_settings(self):
        p, rows = self.rows()
        text = columns.render_data_dictionary(p)
        for name in rows[0]:
            self.assertIn(f"\n{name}\n", text, msg=name)
        self.assertIn("THIS RUN", text)
        self.assertIn("State income tax: IN", text)
        self.assertIn("provisional_income", text)

    def test_numbers_quoted_in_the_text_come_from_the_tax_tables(self):
        text = columns.render_data_dictionary()
        self.assertIn(f"${taxes.STANDARD_DEDUCTION['mfj']:,.0f}", text)
        self.assertIn("2.95% (2026)", text)

    @slow
    def test_run_py_writes_the_dictionary_next_to_the_csvs(self):
        import subprocess, sys
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "run.py", "--config", "config.template.toml", "--out", out],
                               cwd=here, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=r.stderr[-500:])
            self.assertEqual(sorted(os.listdir(out)), ["curve_path.csv", "data_dictionary.txt", "flat_path.csv"])
            with open(os.path.join(out, "data_dictionary.txt"), encoding="utf-8") as f:
                self.assertIn("tax_state", f.read())


# ===========================================================================
# Health insurance: marketplace, Medicare, IRMAA (v6)
# ===========================================================================
class HealthTableTests(unittest.TestCase):
    def test_surcharge_tables_match_the_cms_totals(self):
        totals = [round(health.PART_B_STANDARD + b, 2) for b in health.PART_B_SURCHARGE]
        self.assertEqual(totals, [284.10, 405.80, 527.50, 649.20, 689.90])
        self.assertEqual(health.PART_D_SURCHARGE, [14.50, 37.50, 60.40, 83.30, 91.00])

    def test_tier_four_single_annual_surcharge(self):
        self.assertAlmostEqual(sum(health.irmaa_surcharges(4)) * 12, 6_355.20, places=2)

    def test_single_thresholds_are_half_the_joint_ones_except_the_top(self):
        j, s_ = health.IRMAA_THRESHOLDS["mfj"], health.IRMAA_THRESHOLDS["single"]
        self.assertEqual([x / 2 for x in j[:4]], s_[:4])
        self.assertEqual((j[4], s_[4]), (750_000.0, 500_000.0))

    def test_tier_boundaries_are_strictly_above(self):
        self.assertEqual(health.irmaa_tier(109_000, "single"), 0)
        self.assertEqual(health.irmaa_tier(109_001, "single"), 1)
        self.assertEqual(health.irmaa_tier(218_000, "mfj"), 0)
        self.assertEqual(health.irmaa_tier(274_001, "mfj"), 2)
        self.assertEqual(health.irmaa_tier(750_001, "mfj"), 5)
        self.assertEqual(health.irmaa_tier(0, "single"), 0)

    def test_thresholds_index_but_the_top_one_is_fixed(self):
        self.assertEqual(health.irmaa_tier(218_001, "single", scale=2.0), 1)        # first threshold doubled
        self.assertEqual(health.irmaa_tier(218_000, "single", scale=2.0), 0)
        # 520k: tier 3 if the top threshold were indexed (500k x 3), tier 5 because it is fixed
        self.assertEqual(health.irmaa_tier(520_000, "single", scale=3.0), 5)

    def test_no_surcharge_in_tier_zero(self):
        self.assertEqual(health.irmaa_surcharges(0), (0.0, 0.0))

    def test_months_around_the_65th_birthday(self):
        b = date(1972, 6, 15)
        self.assertEqual([rules.health_months(b, None, y) for y in (2036, 2037, 2038)],
                         [(12, 0), (5, 7), (0, 12)])

    def test_months_stop_the_month_before_death(self):
        b = date(1972, 6, 15)
        self.assertEqual([rules.health_months(b, date(2040, 3, 1), y) for y in (2039, 2040, 2041)],
                         [(0, 12), (0, 2), (0, 0)])
        self.assertEqual(rules.health_months(b, date(2030, 3, 1), 2030), (2, 0))      # dies before 65

    def test_already_on_medicare_at_the_start(self):
        self.assertEqual(rules.health_months(date(1950, 1, 1), None, 2027), (0, 12))


class HealthEngineTests(unittest.TestCase):
    """No inflation and no growth unless stated, so amounts are exact. Plenty of cash."""

    def base(self, **kw):
        d = dict(person1_birth=date(1972, 6, 15), cash=10_000_000.0, nominal_return=0.0, inflation=0.0,
                 health_enabled=True, health_marketplace_monthly=1_500.0)
        d.update(kw)
        return make_params(**d)

    def test_health_off_means_no_health_costs(self):
        rows = simulate(make_params(roth1=1e6), 2_000.0).rows
        self.assertTrue(all(r["health_total"] == 0 and r["need_nominal"] == r["spend_nominal"] for r in rows))

    def test_marketplace_premium_before_65(self):
        row = simulate(self.base(), 2_000.0).rows[0]
        self.assertEqual((row["marketplace_person_months"], row["medicare_person_months"]), (12, 0))
        self.assertAlmostEqual(row["health_marketplace"], 12 * 1_500.0)
        self.assertEqual((row["health_medicare"], row["health_irmaa"]), (0.0, 0.0))
        self.assertAlmostEqual(row["need_nominal"], row["spend_nominal"] + 18_000.0)

    def test_a_couple_pays_twice(self):
        row = simulate(self.base(person2_birth=date(1972, 9, 2)), 2_000.0).rows[0]
        self.assertEqual(row["marketplace_person_months"], 24)
        self.assertAlmostEqual(row["health_marketplace"], 36_000.0)

    def test_medicare_starts_in_the_month_of_the_65th_birthday(self):
        by_year = {r["year"]: r for r in simulate(self.base(), 2_000.0).rows}
        self.assertAlmostEqual(by_year[2036]["health_marketplace"], 18_000.0)
        self.assertAlmostEqual(by_year[2037]["health_marketplace"], 5 * 1_500.0)        # Jan-May
        self.assertAlmostEqual(by_year[2037]["health_medicare"], 7 * health.PART_B_STANDARD, places=2)  # Jun-Dec
        self.assertAlmostEqual(by_year[2038]["health_medicare"], 12 * health.PART_B_STANDARD, places=2)
        self.assertEqual(by_year[2038]["health_marketplace"], 0.0)

    def test_health_premiums_are_paid_from_after_tax_money(self):
        for r in simulate(self.base(trad1=2_000_000.0, person2_birth=date(1972, 9, 2)), 5_000.0).rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"] + r["shortfall_nominal"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)
            self.assertAlmostEqual(r["need_nominal"], r["spend_nominal"] + r["health_total"], delta=0.02)

    def test_health_costs_lower_w(self):
        p = self.base(cash=0.0, roth1=1_500_000.0, nominal_return=0.05, inflation=0.025)
        off = replace(p, health_enabled=False)
        self.assertLess(solve_max_w(p), solve_max_w(off) - 100.0)

    def test_premium_amounts_grow_with_inflation(self):
        p = self.base(inflation=0.025)
        by_year = {r["year"]: r for r in simulate(p, 2_000.0).rows}
        self.assertAlmostEqual(by_year[2030]["health_marketplace"], 18_000.0 * 1.025 ** 3, places=1)
        # Part B is a 2026 amount, so it grows from 2026, not from the start year
        self.assertAlmostEqual(by_year[2040]["health_medicare"], 12 * health.PART_B_STANDARD * 1.025 ** 14, places=1)

    def test_premium_growth_above_inflation(self):
        by_year = {r["year"]: r for r in simulate(self.base(health_premium_growth=0.03), 2_000.0).rows}
        self.assertAlmostEqual(by_year[2032]["health_marketplace"], 18_000.0 * 1.03 ** 5, places=1)

    def test_other_medicare_premiums_are_added(self):
        p = self.base(person1_birth=date(1950, 1, 1), health_medicare_other_monthly=300.0)
        row = simulate(p, 2_000.0).rows[0]
        self.assertAlmostEqual(row["health_medicare"], 12 * (health.PART_B_STANDARD + 300.0), places=2)

    def test_premiums_stop_in_the_death_month(self):
        p = self.base(person1_death=date(2040, 3, 1), person2_birth=date(1972, 9, 2),
                      person2_death=date(2060, 1, 1))
        row = [r for r in simulate(p, 2_000.0).rows if r["year"] == 2040][0]
        self.assertEqual(row["medicare_person_months"], 2 + 12)       # person 1: Jan-Feb; person 2: all year

    # ---- IRMAA ----
    def medicare_household(self, **kw):
        d = dict(person1_birth=date(1950, 1, 1), trad1=3_000_000.0, filing_status="single")
        d.update(kw)
        return self.base(**d)

    def test_irmaa_uses_prior_magi_then_the_models_own_history_two_years_back(self):
        p = self.medicare_household(health_prior_magi=300_000.0)
        by_year = {r["year"]: r for r in simulate(p, 4_000.0).rows}
        self.assertEqual(by_year[2027]["irmaa_lookback_magi"], 300_000.0)
        self.assertEqual(by_year[2028]["irmaa_lookback_magi"], 300_000.0)
        self.assertAlmostEqual(by_year[2029]["irmaa_lookback_magi"], by_year[2027]["agi"], delta=0.01)
        self.assertAlmostEqual(by_year[2030]["irmaa_lookback_magi"], by_year[2028]["agi"], delta=0.01)

    def test_irmaa_amount_for_a_single_person_in_tier_four(self):
        by_year = {r["year"]: r for r in simulate(self.medicare_household(health_prior_magi=300_000.0), 4_000.0).rows}
        self.assertEqual(by_year[2027]["irmaa_tier"], 4)                      # 300k is in 205k-500k (single)
        self.assertAlmostEqual(by_year[2027]["health_irmaa"], 12 * (446.30 + 83.30), places=2)

    def test_the_tier_follows_the_lookback_income(self):
        by_year = {r["year"]: r for r in simulate(self.medicare_household(health_prior_magi=300_000.0), 4_000.0).rows}
        for year in (2029, 2030, 2031):
            magi = by_year[year]["irmaa_lookback_magi"]
            self.assertEqual(by_year[year]["irmaa_tier"], health.irmaa_tier(magi, "single"), msg=str(year))

    def test_each_person_on_medicare_pays_the_surcharge(self):
        p = self.base(person1_birth=date(1950, 1, 1), person2_birth=date(1951, 1, 1), trad1=2_000_000.0,
                      health_prior_magi=300_000.0)                              # joint: tier 2 (274k-342k)
        row = simulate(p, 4_000.0).rows[0]
        self.assertEqual((row["irmaa_tier"], row["medicare_person_months"]), (2, 24))
        self.assertAlmostEqual(row["health_irmaa"], 24 * (202.90 + 37.50), places=2)

    def test_no_irmaa_before_anyone_is_on_medicare(self):
        row = simulate(self.base(health_prior_magi=900_000.0), 2_000.0).rows[0]
        self.assertEqual((row["irmaa_tier"], row["health_irmaa"], row["irmaa_lookback_magi"]), (0, 0.0, 0.0))

    def test_survivors_face_single_thresholds_on_the_couples_old_joint_income(self):
        # The widow's trap: person 1 dies in 2027 (joint return that year); from 2028 the survivor files
        # single. The same 250k lookback income is tier 1 against joint thresholds, tier 4 against single.
        p = make_params(person1_birth=date(1950, 1, 1), person2_birth=date(1952, 1, 1),
                        person1_death=date(2027, 12, 1), person2_death=date(2045, 1, 1), trad1=2_000_000.0,
                        nominal_return=0.0, inflation=0.0, health_enabled=True, health_prior_magi=250_000.0)
        by_year = {r["year"]: r for r in simulate(p, 4_000.0).rows}
        self.assertEqual((by_year[2027]["filing_status"], by_year[2027]["irmaa_tier"]), ("mfj", 1))
        self.assertEqual((by_year[2028]["filing_status"], by_year[2028]["irmaa_tier"]), ("single", 4))
        self.assertEqual(by_year[2028]["irmaa_lookback_magi"], 250_000.0)

    def test_health_costs_helper_uses_the_given_status_and_scale(self):
        p = self.base(person1_birth=date(1950, 1, 1))
        joint = model._health_costs(p, 2030, 3, "mfj", 1.0, 150_000.0)
        single = model._health_costs(p, 2030, 3, "single", 1.0, 150_000.0)
        self.assertEqual((joint["tier"], single["tier"]), (0, 2))
        indexed = model._health_costs(p, 2030, 3, "single", 2.0, 150_000.0)     # thresholds doubled
        self.assertEqual(indexed["tier"], 0)

    def test_tax_disabled_gives_no_irmaa_from_modeled_income(self):
        row = simulate(self.medicare_household(tax_enabled=False), 4_000.0).rows[3]
        self.assertEqual(row["irmaa_tier"], 0)            # modeled AGI is zero when taxes are off


class HealthConfigTests(unittest.TestCase):
    def load(self, block=""):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(ConfigLoadingTests.MINIMAL.format(claims=block))
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    def test_no_block_means_no_health_costs(self):
        self.assertFalse(self.load().health_enabled)

    def test_block_turns_it_on_and_reads_the_values(self):
        p = self.load("[healthcare]\nmarketplace_monthly_per_person = 1500\n"
                      "medicare_other_monthly_per_person = 250\npremium_growth_above_inflation = 0.015\n"
                      "prior_year_magi = 120000")
        self.assertEqual((p.health_enabled, p.health_marketplace_monthly, p.health_medicare_other_monthly,
                          p.health_premium_growth, p.health_prior_magi), (True, 1500.0, 250.0, 0.015, 120_000.0))

    def test_empty_block_still_turns_on_medicare_costs(self):
        p = self.load("[healthcare]")
        self.assertEqual((p.health_enabled, p.health_marketplace_monthly), (True, 0.0))

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.load("[healthcare]\nmarketplace_monthly_per_person = -5")
        with self.assertRaises(ValueError):
            self.load("[healthcare]\npremium_growth_above_inflation = -1.0")


# ===========================================================================
# Inheritances and inherited IRAs (v7)
# ===========================================================================
def make_inh(parent="father", death=date(2040, 3, 1), birth=date(1948, 5, 10), ira=500_000.0,
             share=1.0, assets=()):
    return Inheritance(parent=parent, death_date=death, birth_date=birth, share=share,
                       ira_value=ira, assets=tuple(assets))


class SingleLifeTableTests(unittest.TestCase):
    def test_table_checkpoints_and_shape(self):
        table = rules.SINGLE_LIFE
        self.assertEqual(len(table), 121)                              # ages 0..120
        for age, want in {0: 84.6, 25: 60.2, 50: 36.2, 55: 31.6, 57: 29.8, 61: 26.2, 65: 22.9,
                          70: 18.8, 75: 14.8, 80: 11.2, 90: 5.7, 100: 2.8, 110: 2.0, 120: 1.0}.items():
            self.assertEqual(table[age], want, msg=f"age {age}")
        divisors = [table[a] for a in range(0, 121)]
        self.assertEqual(divisors, sorted(divisors, reverse=True))     # never increases with age
        self.assertEqual(rules.single_life_factor(130), 1.0)
        with self.assertRaises(ValueError):
            rules.single_life_factor(-1)

    def test_irs_worked_example_for_a_57_year_old_beneficiary(self):
        # Pub 590-B style example: $400,000, age 57 -> factor 29.8 -> $13,423; year 2 uses 28.8
        self.assertEqual(round(400_000 / rules.single_life_factor(57)), 13_423)
        self.assertAlmostEqual(405_000 / (rules.single_life_factor(57) - 1), 14_062.5)


class InheritanceValidationTests(unittest.TestCase):
    def test_parent_must_be_father_or_mother(self):
        with self.assertRaises(ValueError):
            Inheritance(parent="uncle", death_date=date(2040, 1, 1))

    def test_share_must_be_in_range(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError, msg=str(bad)):
                make_inh(share=bad)

    def test_ira_needs_the_parents_birth_date(self):
        with self.assertRaises(ValueError):
            Inheritance(parent="father", death_date=date(2040, 1, 1), ira_value=100_000.0)
        Inheritance(parent="father", death_date=date(2040, 1, 1), ira_value=0.0)      # assets only: fine

    def test_parent_must_have_been_taking_rmds(self):
        with self.assertRaises(ValueError):
            make_inh(birth=date(1980, 1, 1), death=date(2040, 1, 1))                 # died at 60
        with self.assertRaises(ValueError):
            make_inh(birth=date(2041, 1, 1))                                          # born after death

    def test_asset_validation(self):
        for kw in (dict(value=-1.0), dict(value=1.0, months_after=-1), dict(value=1.0, cost_pct=1.0),
                   dict(value=1.0, cost_pct=-0.1)):
            with self.assertRaises(ValueError, msg=str(kw)):
                Asset(label="x", **kw)

    def test_params_level_validation(self):
        with self.assertRaises(ValueError):
            make_params(inheritances=(make_inh(), make_inh()))                        # same parent twice
        with self.assertRaises(ValueError):
            make_params(inheritances=(make_inh(death=date(2026, 6, 1)),))             # before start_year
        with self.assertRaises(ValueError):
            make_params(inherited_ira_policy="fast")

    def test_buckets_are_per_parent(self):
        self.assertEqual((make_inh("father").bucket, make_inh("mother").bucket), ("inh_f", "inh_m"))


class InheritanceEngineTests(unittest.TestCase):
    """Zero growth, zero inflation, no tax unless stated; plenty of cash covers spending, so inherited
    withdrawals are exactly the forced (required) amounts. Heir (person 1) born 1972; father dies Mar 2040,
    so the heir is 69 in 2041 (Single Life factor 19.6) and the account must be empty by Dec 31, 2050."""

    def base(self, **kw):
        d = dict(person1_birth=date(1972, 6, 15), cash=10_000_000.0, nominal_return=0.0, inflation=0.0,
                 tax_enabled=False, inheritances=(make_inh(),))
        d.update(kw)
        return make_params(**d)

    def by_year(self, p, w=2_000.0):
        return {r["year"]: r for r in simulate(p, w).rows}

    def test_without_inheritances_every_inheritance_column_is_zero(self):
        rows = simulate(make_params(roth1=1e6), 2_000.0).rows
        for key in ("inheritance_ira_received", "inheritance_cash_received", "inherited_ira_required",
                    "draw_inherited_ira", "end_inh_f", "end_inh_m"):
            self.assertTrue(all(r[key] == 0.0 for r in rows), msg=key)

    def test_ira_arrives_at_the_start_of_the_year_of_death(self):
        rows = self.by_year(self.base())
        self.assertEqual(rows[2039]["inheritance_ira_received"], 0.0)
        self.assertEqual(rows[2039]["end_inh_f"], 0.0)
        self.assertAlmostEqual(rows[2040]["inheritance_ira_received"], 500_000.0)
        self.assertEqual(rows[2041]["inheritance_ira_received"], 0.0)

    def test_year_of_death_requires_the_parents_own_rmd(self):
        row = self.by_year(self.base())[2040]
        self.assertAlmostEqual(row["inherited_ira_required"], 500_000.0 / rules.rmd_divisor(2040 - 1948), places=2)
        self.assertAlmostEqual(row["draw_inherited_ira"], row["inherited_ira_required"], places=2)

    def test_min_policy_follows_the_single_life_schedule_then_empties_in_year_ten(self):
        rows = self.by_year(self.base(inherited_ira_policy="min"))
        bal = 500_000.0 - rows[2040]["inherited_ira_required"]
        for year in range(2041, 2050):                                       # years 1-9
            divisor = rules.single_life_factor(2041 - 1972) - (year - 2041)  # 19.6, 18.6, ...
            due = bal / divisor
            self.assertAlmostEqual(rows[year]["inherited_ira_required"], due, delta=0.01, msg=str(year))
            bal -= due
        self.assertAlmostEqual(rows[2050]["inherited_ira_required"], bal, delta=0.01)   # year 10: everything
        self.assertEqual(rows[2050]["end_inh_f"], 0.0)

    def test_even_policy_takes_level_draws_over_the_window(self):
        rows = self.by_year(self.base(inherited_ira_policy="even"))
        level = (500_000.0 - rows[2040]["inherited_ira_required"]) / 10
        for year in range(2041, 2051):
            self.assertAlmostEqual(rows[year]["draw_inherited_ira"], level, delta=0.01, msg=str(year))
        self.assertEqual(rows[2050]["end_inh_f"], 0.0)

    def test_every_policy_empties_the_account_by_the_deadline_even_with_growth(self):
        for policy in ("even", "min"):
            for ret in (0.0, 0.05):
                rows = self.by_year(self.base(inherited_ira_policy=policy, nominal_return=ret))
                self.assertEqual(rows[2050]["end_inh_f"], 0.0, msg=f"{policy} {ret}")
                self.assertEqual(rows[2052]["end_inh_f"], 0.0, msg=f"{policy} {ret}")
                self.assertGreater(rows[2049]["end_inh_f"], 0.0, msg=f"{policy} {ret}")

    def test_min_policy_defers_more_than_even_until_the_final_year(self):
        even = self.by_year(self.base(inherited_ira_policy="even"))
        minimum = self.by_year(self.base(inherited_ira_policy="min"))
        self.assertGreater(minimum[2049]["end_inh_f"], even[2049]["end_inh_f"])
        self.assertGreater(minimum[2050]["draw_inherited_ira"], 5 * even[2050]["draw_inherited_ira"])

    def test_inherited_ira_income_is_taxed_with_no_early_withdrawal_penalty(self):
        # heir born 1985 is only 56 in 2041: their own accounts would be locked, the inherited IRA is not
        p = self.base(person1_birth=date(1985, 1, 1), tax_enabled=True, trad1=1_000_000.0)
        row = self.by_year(p)[2041]
        self.assertGreater(row["draw_inherited_ira"], 0.0)
        self.assertEqual(row["penalty_early_withdrawal"], 0.0)
        self.assertGreaterEqual(row["agi"], row["draw_inherited_ira"] - 0.01)
        self.assertAlmostEqual(row["tax_federal"], taxes.federal_tax(row["draw_inherited_ira"], 0.0, "mfj"), delta=0.01)
        self.assertEqual(row["draw_trad"], 0.0)                                # own traditional untouched

    def test_assets_arrive_in_the_year_of_the_sale_net_of_costs(self):
        assets = (Asset("house", 300_000.0, months_after=9, cost_pct=0.10),     # Mar 2040 + 9 months = Dec 2040
                  Asset("land", 100_000.0, months_after=12, cost_pct=0.0))      # Mar 2041
        rows = self.by_year(self.base(inheritances=(make_inh(ira=0.0, assets=assets),)))
        self.assertAlmostEqual(rows[2040]["inheritance_cash_received"], 270_000.0)
        self.assertAlmostEqual(rows[2041]["inheritance_cash_received"], 100_000.0)
        self.assertEqual(rows[2042]["inheritance_cash_received"], 0.0)
        self.assertAlmostEqual(rows[2041]["end_taxable"], 370_000.0)          # nothing taken out: cash pays the bills

    def test_amounts_are_in_todays_dollars_and_scale_with_inflation(self):
        p = self.base(inflation=0.025, inheritances=(make_inh(ira=0.0, assets=(Asset("cash", 100_000.0),)),))
        self.assertAlmostEqual(self.by_year(p)[2040]["inheritance_cash_received"], 100_000.0 * 1.025 ** 13, places=1)

    def test_the_ira_amount_also_scales_with_inflation(self):
        p = self.base(inflation=0.025, inheritances=(make_inh(ira=500_000.0),))
        self.assertAlmostEqual(self.by_year(p)[2040]["inheritance_ira_received"], 500_000.0 * 1.025 ** 13, places=1)

    def test_sale_proceeds_have_full_basis_so_selling_them_realizes_no_gain(self):
        p = self.base(cash=1_000_000.0, tax_enabled=True, fixed_order=("taxable", "cash"),
                      inheritances=(make_inh(ira=0.0, assets=(Asset("house", 300_000.0),)),))
        rows = self.by_year(p, 5_000.0)
        for year in (2040, 2041, 2042):
            self.assertGreater(rows[year]["draw_taxable"], 0.0, msg=str(year))
            self.assertEqual(rows[year]["realized_gains"], 0.0, msg=str(year))

    def test_share_scales_the_ira_and_the_assets(self):
        ev = make_inh(share=0.5, assets=(Asset("house", 300_000.0, cost_pct=0.1),))
        row = self.by_year(self.base(inheritances=(ev,)))[2040]
        self.assertAlmostEqual(row["inheritance_ira_received"], 250_000.0)
        self.assertAlmostEqual(row["inheritance_cash_received"], 135_000.0)

    def test_an_heir_who_dies_first_receives_nothing(self):
        p = self.base(person2_birth=date(1972, 9, 2), person1_death=date(2035, 6, 1))
        rows = self.by_year(p)
        self.assertTrue(all(r["inheritance_ira_received"] == 0.0 and r["end_inh_f"] == 0.0 for r in rows.values()))

    def test_two_parents_have_separate_windows_and_buckets(self):
        p = self.base(inheritances=(make_inh("father", date(2040, 3, 1)),
                                    make_inh("mother", date(2043, 9, 1), birth=date(1950, 1, 1), ira=300_000.0)))
        rows = self.by_year(p)
        self.assertEqual(rows[2050]["end_inh_f"], 0.0)                         # father's deadline
        self.assertGreater(rows[2050]["end_inh_m"], 0.0)                       # mother's still open
        self.assertEqual(rows[2053]["end_inh_m"], 0.0)
        self.assertAlmostEqual(rows[2045]["inherited_ira_required"],
                               rows[2045]["draw_inherited_ira"], delta=0.01)

    def test_inheritances_raise_w(self):
        p = make_params(person1_birth=date(1972, 6, 15), roth1=1_000_000.0, nominal_return=0.05, inflation=0.025,
                        inheritances=(make_inh(assets=(Asset("house", 300_000.0, months_after=6),)),))
        self.assertGreater(solve_max_w(p), solve_max_w(replace(p, inheritances=())) + 100.0)

    def test_inherited_distributions_count_against_the_standard_deduction_cap(self):
        # trad_std_then_cash fills one standard deduction with traditional money. Inherited draws above
        # that deduction already do the job, so no extra draw from your own traditional account.
        p = self.base(trad1=1_000_000.0, strategy="trad_std_then_cash", inheritances=(make_inh(ira=1_500_000.0),))
        rows = self.by_year(p, 4_000.0)
        for year in (2041, 2042, 2043):
            self.assertGreater(rows[year]["draw_inherited_ira"], 32_200.0)
            self.assertEqual(rows[year]["draw_trad"], 0.0, msg=str(year))

    def test_a_shortfall_taps_the_inherited_ira_before_your_own_traditional_accounts(self):
        # No cash or taxable money. Person 2 (born 1960, unlocked) has $1M of their own traditional money.
        # The father dies in 2028; with the "min" policy the required amount is far below the $60k need,
        # so the extra must come from the inherited IRA FIRST while the own account stays untouched.
        p = make_params(person1_birth=date(1972, 6, 15), person2_birth=date(1960, 1, 1), trad2=1_000_000.0,
                        nominal_return=0.0, inflation=0.0, tax_enabled=False, inherited_ira_policy="min",
                        inheritances=(make_inh(death=date(2028, 3, 1), birth=date(1940, 1, 1), ira=500_000.0),))
        rows = {r["year"]: r for r in simulate(p, 5_000.0).rows}
        row = rows[2029]                                     # need 60,000; the required minimum is ~16,000
        self.assertGreater(row["inherited_ira_required"], 0.0)
        self.assertLess(row["inherited_ira_required"], 30_000.0)
        self.assertAlmostEqual(row["draw_inherited_ira"], 60_000.0, delta=0.01)       # tapped beyond the minimum
        self.assertEqual(row["draw_trad"], 0.0)                                       # own account untouched
        # the only own-account draw was in 2027, before the inheritance arrived
        self.assertAlmostEqual(row["end_trad2"], 1_000_000.0 - rows[2027]["draw_trad"], delta=0.01)

    def test_total_balance_includes_inherited_buckets(self):
        for r in self.by_year(self.base(trad1=300_000.0, roth2=0.0)).values():
            parts = (r["end_cash"] + r["end_taxable"] + r["end_trad1"] + r["end_trad2"] + r["end_roth1"]
                     + r["end_roth2"] + r["end_inh_f"] + r["end_inh_m"])
            self.assertAlmostEqual(parts, r["end_total_nominal"], delta=0.05)

    def test_money_flows_balance_in_inheritance_years(self):
        p = self.base(cash=500_000.0, tax_enabled=True,
                      inheritances=(make_inh(assets=(Asset("house", 300_000.0, months_after=6, cost_pct=0.07),)),))
        for r in self.by_year(p, 6_000.0).values():
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"] + r["shortfall_nominal"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)


class InheritanceConfigTests(unittest.TestCase):
    def load(self, block=""):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(ConfigLoadingTests.MINIMAL.format(claims=block))
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    FATHER = ("[inheritance.father]\ndeath_date = 2040-03-01\nbirth_date = 1948-05-10\nshare = 0.5\n"
              "ira_value = 700000\n[[inheritance.father.assets]]\nlabel = \"House\"\nvalue = 350000\n"
              "sale_months_after_death = 9\nsale_cost_pct = 0.07\n")

    def test_no_block_means_no_inheritance(self):
        p = self.load()
        self.assertEqual((p.inheritances, p.inherited_ira_policy), ((), "even"))

    def test_father_block_with_assets(self):
        ev = self.load(self.FATHER).inheritances[0]
        self.assertEqual((ev.parent, ev.death_date, ev.birth_date, ev.share, ev.ira_value),
                         ("father", date(2040, 3, 1), date(1948, 5, 10), 0.5, 700_000.0))
        self.assertEqual(ev.assets, (Asset("House", 350_000.0, 9, 0.07),))

    def test_both_parents_and_the_policy(self):
        p = self.load("[inheritance]\nira_withdrawal = \"min\"\n" + self.FATHER +
                      "[inheritance.mother]\ndeath_date = 2046-11-15\nbirth_date = 1950-08-20\nira_value = 500000\n")
        self.assertEqual([e.parent for e in p.inheritances], ["father", "mother"])
        self.assertEqual(p.inherited_ira_policy, "min")

    def test_a_block_without_a_death_date_is_an_error(self):
        with self.assertRaises(ValueError):
            self.load("[inheritance.father]\nira_value = 100000\nbirth_date = 1948-05-10")

    def test_bad_values_are_errors(self):
        with self.assertRaises(ValueError):
            self.load("[inheritance]\nira_withdrawal = \"fast\"")
        with self.assertRaises(ValueError):
            self.load("[inheritance.father]\ndeath_date = \"2040-03-01\"")             # a string, not a date
        with self.assertRaises(ValueError):
            self.load("[inheritance.father]\ndeath_date = 2040-03-01\nira_value = 5\n")   # IRA but no birth date


# ===========================================================================
# Spousal benefit (v8)
# ===========================================================================
def _spousal_reduction(months_early):
    """Independent copy of the rule: 25/36 of 1% a month for 36 months, then 5/12 of 1% a month."""
    return months_early * 25 / 3600 if months_early <= 36 else 36 * 25 / 3600 + (months_early - 36) * 5 / 1200


def _own_factor(claim_months, fra=804):
    """Independent copy of the own-benefit adjustment (FRA 67): 5/9% then 5/12% early, 2/3% a month late."""
    if claim_months >= fra:
        return 1 + (claim_months - fra) * (2 / 3) / 100
    early = fra - claim_months
    return 1 - (min(early, 36) * (5 / 9) + max(0, early - 36) * (5 / 12)) / 100


class SpousalBenefitRuleTests(unittest.TestCase):
    H = date(1965, 1, 1)          # both people have FRA 67 (804 months)

    def monthly(self, persons, year, spousal=True):
        total, top_up = rules.ss_real_by_year_detail(persons, year, year, spousal)[year]
        return total / 12, top_up / 12

    def test_both_claim_at_fra_the_lower_earner_gets_the_excess(self):
        total, top_up = self.monthly([(self.H, 3000, None, None), (self.H, 1000, None, None)], 2033)
        self.assertAlmostEqual(top_up, 500.0)                         # 50% x 3,000 - 1,000
        self.assertAlmostEqual(total, 3000 + 1000 + 500)

    def test_no_top_up_when_the_lower_pia_is_at_least_half(self):
        for low in (1600, 1800, 3000):
            _, top_up = self.monthly([(self.H, 3200, None, None), (self.H, low, None, None)], 2040)
            self.assertEqual(top_up, 0.0, msg=str(low))

    def test_the_roles_do_not_depend_on_the_order_of_the_list(self):
        a = self.monthly([(self.H, 3000, None, None), (self.H, 1000, None, None)], 2040)
        b = self.monthly([(self.H, 1000, None, None), (self.H, 3000, None, None)], 2040)
        self.assertEqual(a, b)

    def test_a_spouse_with_no_own_benefit_gets_half_of_the_others(self):
        total, top_up = self.monthly([(self.H, 3000, None, None), (self.H, 0, None, None)], 2040)
        self.assertAlmostEqual(top_up, 1500.0)
        self.assertAlmostEqual(total, 3000 + 1500)

    def test_it_is_based_on_the_pia_not_the_delayed_check(self):
        total, top_up = self.monthly([(self.H, 3000, 70 * 12, None), (self.H, 1000, None, None)], 2040)
        self.assertAlmostEqual(top_up, 500.0)                         # not 50% of 3,720
        self.assertAlmostEqual(total, 3000 * 1.24 + 1000 + 500)

    def test_it_waits_for_the_higher_earner_to_file(self):
        persons = [(self.H, 3000, 67 * 12, None), (self.H, 1000, 62 * 12, None)]      # L claims at 62, H at 67
        before = self.monthly(persons, 2031)                           # H has not filed yet
        after = self.monthly(persons, 2033)
        self.assertEqual(before[1], 0.0)
        self.assertAlmostEqual(before[0], 1000 * 0.70)                 # L's own reduced check only
        self.assertAlmostEqual(after[1], 500.0)                        # by then L is 67: no reduction
        self.assertAlmostEqual(after[0], 3000 + 700 + 500)

    def test_starting_before_fra_is_reduced(self):
        for claim_years, factor in ((62, 0.65), (64, 0.75), (66, 1 - 12 * 25 / 3600)):
            _, top_up = self.monthly([(self.H, 3000, claim_years * 12, None),
                                      (self.H, 1000, claim_years * 12, None)], 2045)
            self.assertAlmostEqual(top_up, 500 * factor, places=6, msg=str(claim_years))

    def test_the_reduction_follows_the_spousal_start_not_the_lower_earners_own_claim(self):
        # L claims at 62 but the top-up only starts when H files at 66: L is then 66 (12 months early)
        _, top_up = self.monthly([(self.H, 3000, 66 * 12, None), (self.H, 1000, 62 * 12, None)], 2045)
        self.assertAlmostEqual(top_up, 500 * (1 - 12 * 25 / 3600), places=6)

    def test_no_credit_for_waiting_past_fra(self):
        _, top_up = self.monthly([(self.H, 3000, 70 * 12, None), (self.H, 1000, 70 * 12, None)], 2045)
        self.assertAlmostEqual(top_up, 500.0)

    def test_switched_off(self):
        total, top_up = self.monthly([(self.H, 3000, None, None), (self.H, 1000, None, None)], 2040, spousal=False)
        self.assertEqual((total, top_up), (4000.0, 0.0))

    def test_a_single_person_has_no_spousal_benefit(self):
        self.assertEqual(self.monthly([(self.H, 3000, None, None)], 2040), (3000.0, 0.0))
        self.assertEqual(rules.spousal_topups([(self.H, 3000, None, None)])[0]["monthly"], 0.0)

    def test_it_stops_at_the_first_death_and_the_survivor_benefit_takes_over(self):
        # H dies Jul 2045: the survivor (L) then gets the larger of own (1,000) and the survivor benefit (3,000)
        persons = [(self.H, 3000, None, date(2045, 7, 15)), (self.H, 1000, None, date(2060, 1, 1))]
        total, top_up = rules.ss_real_by_year_detail(persons, 2045, 2046)[2045]
        self.assertAlmostEqual(total, 6 * 4500 + 6 * 3000)            # spousal only before the death
        self.assertAlmostEqual(top_up, 6 * 500)
        self.assertAlmostEqual(rules.ss_real_by_year_detail(persons, 2046, 2046)[2046][1], 0.0)
        # L dies first: H alone, own benefit only
        persons = [(self.H, 3000, None, date(2060, 1, 1)), (self.H, 1000, None, date(2045, 7, 15))]
        total, top_up = rules.ss_real_by_year_detail(persons, 2045, 2046)[2046]
        self.assertAlmostEqual((total, top_up), (12 * 3000, 0.0))

    def test_every_combination_of_claim_ages_matches_an_independent_calculation(self):
        # The scenario-grid case: both people claim at every age from 62 to 70 (whole years plus a
        # few odd months). H born 1965 (PIA 3,200); L born 1967 (PIA 1,100). Check a year when both
        # are collecting and alive against arithmetic that does not use the rules.py helpers.
        h_birth, l_birth = date(1965, 1, 1), date(1967, 1, 1)
        h_pia, l_pia = 3_200.0, 1_100.0
        ages = [a * 12 for a in range(62, 71)] + [62 * 12 + 5, 66 * 12 + 7, 69 * 12 + 11]
        count = 0
        for ch in ages:
            for cl in ages:
                persons = [(h_birth, h_pia, ch, None), (l_birth, l_pia, cl, None)]
                total, top_up = rules.ss_real_by_year_detail(persons, 2050, 2050)[2050]
                start = max(1965 * 12 + ch, 1967 * 12 + cl)                        # both have filed
                early = max(0, (1967 * 12 + 804) - start)                          # months before L's FRA
                excess = max(0.0, 0.5 * h_pia - l_pia)
                spousal = excess * (1 - _spousal_reduction(early))
                want = 12 * (h_pia * _own_factor(ch) + l_pia * _own_factor(cl) + spousal)
                self.assertAlmostEqual(total, want, places=4, msg=f"H{ch} L{cl}")
                self.assertAlmostEqual(top_up, 12 * spousal, places=4, msg=f"H{ch} L{cl}")
                self.assertGreaterEqual(top_up, 0.0)
                self.assertLessEqual(spousal, excess + 1e-9)
                count += 1
        self.assertEqual(count, len(ages) ** 2)

    def test_delaying_the_higher_earner_never_starts_the_top_up_earlier(self):
        l_claim = 64 * 12
        starts = [rules.spousal_topups([(self.H, 3200, h * 12, None), (self.H, 1100, l_claim, None)])[1]["start"]
                  for h in range(62, 71)]
        self.assertEqual(starts, sorted(starts))


class SpousalBenefitEngineTests(unittest.TestCase):
    def couple(self, **kw):
        d = dict(person1_birth=date(1965, 1, 1), person2_birth=date(1965, 1, 1), roth1=3_000_000.0,
                 nominal_return=0.0, inflation=0.0, tax_enabled=False, ss_fra_monthly1=3_000.0,
                 ss_fra_monthly2=1_000.0)
        d.update(kw)
        return make_params(**d)

    def test_the_column_is_part_of_ss_gross_and_starts_when_both_have_claimed(self):
        rows = {r["year"]: r for r in simulate(self.couple(), 3_000.0).rows}
        self.assertEqual(rows[2031]["ss_spousal"], 0.0)
        self.assertAlmostEqual(rows[2032]["ss_spousal"], 12 * 500.0)
        self.assertAlmostEqual(rows[2032]["ss_gross"], 12 * 4_500.0)

    def test_no_column_value_without_an_excess(self):
        rows = simulate(self.couple(ss_fra_monthly2=1_800.0), 3_000.0).rows
        self.assertTrue(all(r["ss_spousal"] == 0.0 for r in rows))

    def test_the_spousal_benefit_raises_w(self):
        p = make_params(person1_birth=date(1965, 1, 1), person2_birth=date(1965, 1, 1), roth1=1_500_000.0,
                        ss_fra_monthly1=3_000.0, ss_fra_monthly2=1_000.0)
        self.assertGreater(solve_max_w(p), solve_max_w(replace(p, ss_spousal=False)) + 100.0)

    def test_delaying_the_higher_earner_delays_the_top_up_in_the_rows(self):
        early = {r["year"]: r for r in simulate(self.couple(ss_claim_months1=62 * 12, ss_claim_months2=62 * 12), 3_000.0).rows}
        late = {r["year"]: r for r in simulate(self.couple(ss_claim_months1=70 * 12, ss_claim_months2=62 * 12), 3_000.0).rows}
        self.assertGreater(early[2028]["ss_spousal"], 0.0)            # both claimed at 62 (Jan 2027)
        self.assertEqual(late[2028]["ss_spousal"], 0.0)               # waits for the higher earner (Jan 2035)
        self.assertAlmostEqual(late[2036]["ss_spousal"], 12 * 500.0)   # started in 2035 with no reduction

    def test_the_benefit_multiplier_scales_the_top_up_too(self):
        full = simulate(self.couple(), 3_000.0).rows[10]["ss_spousal"]
        cut = simulate(self.couple(ss_multiplier=0.77), 3_000.0).rows[10]["ss_spousal"]
        self.assertAlmostEqual(cut, 0.77 * full, places=2)

    def test_the_switch_is_read_from_the_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(ConfigLoadingTests.MINIMAL.format(claims="spousal_benefit = false"))
            path = f.name
        try:
            self.assertFalse(model.load_params(path).ss_spousal)
        finally:
            os.remove(path)
        self.assertTrue(make_params().ss_spousal)                     # on by default


# ===========================================================================
# Asset classes and yearly tax on dividends and interest (v8)
# ===========================================================================
CLASSES = (AssetClass("stocks", 0.07, 0.015, 0.95), AssetClass("bonds", 0.04, 0.04, 0.0),
           AssetClass("cash", 0.03, 0.03, 0.0))


def mix(cash=None, taxable=None, traditional=None, roth=None, inherited=None):
    """An [allocation]: one {class: weight} per account group (defaults: cash 100% cash, the rest 100% stocks)."""
    groups = {"cash": cash or {"cash": 1.0}, "taxable": taxable or {"stocks": 1.0},
              "traditional": traditional or {"stocks": 1.0}, "roth": roth or {"stocks": 1.0},
              "inherited": inherited or {"stocks": 1.0}}
    return tuple((g, tuple(m.items())) for g, m in groups.items())


class AssetClassValidationTests(unittest.TestCase):
    def test_class_validation(self):
        for kw in (dict(name=""), dict(name="x", total_return=-1.0), dict(name="x", total_return=0.05, income_yield=-0.01),
                   dict(name="x", total_return=0.05, qualified_share=1.5)):
            kw.setdefault("total_return", 0.05)
            with self.assertRaises(ValueError, msg=str(kw)):
                AssetClass(**kw)

    def test_names_must_be_unique(self):
        with self.assertRaises(ValueError):
            make_params(asset_classes=CLASSES + (AssetClass("bonds", 0.05),), allocations=mix())

    def test_every_account_group_needs_a_mix(self):
        partial = mix()[:-1]                                           # no "inherited"
        with self.assertRaises(ValueError):
            make_params(asset_classes=CLASSES, allocations=partial)

    def test_weights_must_add_to_one_and_be_non_negative(self):
        for bad in ({"stocks": 0.6, "bonds": 0.3}, {"stocks": 1.2, "bonds": -0.2}):
            with self.assertRaises(ValueError, msg=str(bad)):
                make_params(asset_classes=CLASSES, allocations=mix(taxable=bad))

    def test_unknown_class_in_a_mix_is_rejected(self):
        with self.assertRaises(ValueError):
            make_params(asset_classes=CLASSES, allocations=mix(taxable={"crypto": 1.0}))

    def test_allocation_without_classes_is_rejected(self):
        with self.assertRaises(ValueError):
            make_params(allocations=mix())


class AssetClassRateTests(unittest.TestCase):
    def test_without_classes_every_account_uses_the_single_rate_and_has_no_income(self):
        rates = model._bucket_rates(make_params(nominal_return=0.05))
        self.assertTrue(all(v == (0.05, 0.0, 0.0) for v in rates.values()))

    def test_weighted_returns_and_yields(self):
        p = make_params(asset_classes=CLASSES, allocations=mix(taxable={"stocks": 0.8, "bonds": 0.1, "cash": 0.1},
                                                              traditional={"stocks": 0.6, "bonds": 0.4}))
        total, ordinary, qualified = model._bucket_rates(p)["taxable"]
        self.assertAlmostEqual(total, 0.8 * 0.07 + 0.1 * 0.04 + 0.1 * 0.03)
        self.assertAlmostEqual(qualified, 0.8 * 0.015 * 0.95)                       # only stocks have qualified income
        self.assertAlmostEqual(ordinary, 0.8 * 0.015 * 0.05 + 0.1 * 0.04 + 0.1 * 0.03)
        self.assertAlmostEqual(model._bucket_rates(p)["trad2"][0], 0.6 * 0.07 + 0.4 * 0.04)   # both people's accounts
        self.assertEqual(model._bucket_rates(p)["trad1"], model._bucket_rates(p)["trad2"])

    def test_inherited_buckets_use_the_inherited_mix(self):
        p = make_params(asset_classes=CLASSES, allocations=mix(inherited={"bonds": 1.0}))
        self.assertEqual(model._bucket_rates(p)["inh_f"][0], 0.04)
        self.assertEqual(model._bucket_rates(p)["inh_m"][0], 0.04)


class AssetClassEngineTests(unittest.TestCase):
    """One person born 1965 (unlocked, no RMD until 2040), inflation 0 so amounts are exact."""

    def base(self, **kw):
        d = dict(person1_birth=date(1965, 1, 1), inflation=0.0, asset_classes=CLASSES, allocations=mix())
        d.update(kw)
        return make_params(**d)

    def test_each_account_grows_at_its_own_weighted_rate(self):
        p = self.base(tax_enabled=False, cash=100_000.0, taxable_value=100_000.0, taxable_basis=100_000.0,
                      trad1=100_000.0, roth1=100_000.0,
                      allocations=mix(cash={"cash": 1.0}, taxable={"stocks": 1.0}, traditional={"bonds": 1.0},
                                      roth={"stocks": 0.5, "bonds": 0.5}))
        row = simulate(p, 0.0).rows[0]
        self.assertAlmostEqual(row["end_cash"], 103_000.0, places=1)
        self.assertAlmostEqual(row["end_taxable"], 107_000.0, places=1)
        self.assertAlmostEqual(row["end_trad1"], 104_000.0, places=1)
        self.assertAlmostEqual(row["end_roth1"], 105_500.0, places=1)

    def test_no_classes_keeps_the_single_rate_and_no_income(self):
        p = make_params(person1_birth=date(1965, 1, 1), inflation=0.0, nominal_return=0.05, taxable_value=100_000.0,
                        taxable_basis=100_000.0, cash=50_000.0)
        rows = simulate(p, 0.0).rows
        self.assertTrue(all(r["inv_income_ordinary"] == 0.0 and r["inv_income_qualified"] == 0.0 for r in rows))
        self.assertAlmostEqual(rows[0]["end_taxable"], 105_000.0, places=1)

    def test_interest_in_the_taxable_account_is_taxed_as_ordinary_income_every_year(self):
        # 2M of bonds (4% interest, all ordinary), nothing else, spending 0: the only cash need is the tax.
        p = self.base(taxable_value=2_000_000.0, taxable_basis=2_000_000.0, allocations=mix(taxable={"bonds": 1.0}))
        row = simulate(p, 0.0).rows[0]
        self.assertAlmostEqual(row["inv_income_ordinary"], (2_000_000.0 - row["draw_taxable"]) * 0.04, delta=0.02)
        self.assertEqual(row["inv_income_qualified"], 0.0)
        self.assertAlmostEqual(row["agi"], row["inv_income_ordinary"], delta=0.02)
        self.assertAlmostEqual(row["tax_federal"], taxes.federal_tax(row["inv_income_ordinary"], 0.0, "mfj"), delta=0.02)
        self.assertGreater(row["draw_taxable"], 0.0)                         # the tax had to be paid from somewhere

    def test_qualified_dividends_are_taxed_at_capital_gains_rates(self):
        qd = AssetClass("qd", 0.06, 0.03, 1.0)
        p = self.base(taxable_value=3_000_000.0, taxable_basis=3_000_000.0, asset_classes=CLASSES + (qd,),
                      allocations=mix(taxable={"qd": 1.0}))
        row = simulate(p, 0.0).rows[0]
        self.assertEqual(row["inv_income_ordinary"], 0.0)
        self.assertAlmostEqual(row["inv_income_qualified"], 3_000_000.0 * 0.03, delta=1.0)      # ~90,000
        self.assertAlmostEqual(row["tax_federal"], taxes.federal_tax(0.0, row["inv_income_qualified"], "mfj"), delta=0.02)
        self.assertEqual(row["tax_federal"], 0.0)                            # inside the 0% bracket (32.2k deduction + 98.9k)

    def test_ordinary_and_qualified_parts_follow_the_qualified_share(self):
        half = AssetClass("half", 0.06, 0.04, 0.5)
        p = self.base(taxable_value=1_000_000.0, taxable_basis=1_000_000.0, asset_classes=CLASSES + (half,),
                      allocations=mix(taxable={"half": 1.0}))
        row = simulate(p, 0.0).rows[0]
        self.assertAlmostEqual(row["inv_income_ordinary"], row["inv_income_qualified"], delta=0.02)
        self.assertAlmostEqual(row["inv_income_ordinary"] + row["inv_income_qualified"],
                               (1_000_000.0 - row["draw_taxable"]) * 0.04, delta=0.02)

    def test_cash_interest_is_taxed_too(self):
        p = self.base(cash=1_000_000.0)
        row = simulate(p, 0.0).rows[0]
        self.assertAlmostEqual(row["inv_income_ordinary"], (1_000_000.0 - row["draw_cash"]) * 0.03, delta=0.02)
        self.assertAlmostEqual(row["agi"], row["inv_income_ordinary"], delta=0.02)

    def test_traditional_roth_and_inherited_accounts_have_no_yearly_income(self):
        p = self.base(trad1=2_000_000.0, roth1=1_000_000.0, allocations=mix(traditional={"bonds": 1.0}, roth={"bonds": 1.0}))
        rows = simulate(p, 0.0).rows[:5]
        self.assertTrue(all(r["inv_income_ordinary"] == 0.0 and r["agi"] == 0.0 and r["tax_federal"] == 0.0 for r in rows))

    def test_reinvested_income_raises_the_cost_basis(self):
        # 1M at 5% with a 2% qualified yield. Year 1 nothing is sold (the income is inside the 0% bracket).
        # Year 2 a one-time purchase forces a sale: gain fraction = 1 - (1.0M + 20k reinvested) / 1.05M.
        qd = AssetClass("qd", 0.05, 0.02, 1.0)
        p = self.base(taxable_value=1_000_000.0, taxable_basis=1_000_000.0, asset_classes=CLASSES + (qd,),
                      allocations=mix(taxable={"qd": 1.0}), fixed_order=("taxable", "cash"),
                      purchases=((2028, 60_000.0, "x"),))
        rows = {r["year"]: r for r in simulate(p, 0.0).rows}
        self.assertEqual(rows[2027]["draw_taxable"], 0.0)
        self.assertAlmostEqual(rows[2027]["end_taxable"], 1_050_000.0, delta=0.05)
        self.assertAlmostEqual(rows[2028]["realized_gains"] / rows[2028]["draw_taxable"], 1 - 1_020_000.0 / 1_050_000.0, places=5)

    def test_only_taxable_account_income_raises_basis_not_cash_interest(self):
        # cash AND taxable both earn income in year 1; only the taxable account's 20k adds to its basis
        qd = AssetClass("qd", 0.05, 0.02, 1.0)
        p = self.base(cash=1_000_000.0, taxable_value=1_000_000.0, taxable_basis=1_000_000.0,
                      asset_classes=CLASSES + (qd,), allocations=mix(taxable={"qd": 1.0}),
                      fixed_order=("taxable", "cash"), purchases=((2028, 60_000.0, "x"),))
        rows = {r["year"]: r for r in simulate(p, 0.0).rows}
        self.assertAlmostEqual(rows[2028]["realized_gains"] / rows[2028]["draw_taxable"], 1 - 1_020_000.0 / 1_050_000.0, places=5)

    def test_yearly_income_tax_lowers_w_compared_with_deferring_the_same_return(self):
        # same 4% total return: paid out as taxable interest vs taken as untaxed price growth
        paid = AssetClass("paid", 0.04, 0.04, 0.0)
        grown = AssetClass("grown", 0.04, 0.0, 0.0)
        common = dict(taxable_value=1_500_000.0, taxable_basis=1_500_000.0, nominal_return=0.04)
        w_paid = solve_max_w(self.base(**common, asset_classes=CLASSES + (paid,), allocations=mix(taxable={"paid": 1.0})))
        w_grown = solve_max_w(self.base(**common, asset_classes=CLASSES + (grown,), allocations=mix(taxable={"grown": 1.0})))
        self.assertLess(w_paid, w_grown - 20.0)

    def test_yields_do_not_matter_when_everything_is_in_a_roth(self):
        paid = AssetClass("paid", 0.04, 0.04, 0.0)
        grown = AssetClass("grown", 0.04, 0.0, 0.0)
        a = self.base(roth1=1_500_000.0, asset_classes=CLASSES + (paid,), allocations=mix(roth={"paid": 1.0}))
        b = self.base(roth1=1_500_000.0, asset_classes=CLASSES + (grown,), allocations=mix(roth={"grown": 1.0}))
        self.assertAlmostEqual(solve_max_w(a), solve_max_w(b), places=2)

    def test_investment_income_feeds_state_tax_and_the_irmaa_lookback(self):
        p = self.base(person1_birth=date(1950, 1, 1), taxable_value=2_000_000.0, taxable_basis=2_000_000.0,
                      state_origin="IN", health_enabled=True, filing_status="single",
                      allocations=mix(taxable={"bonds": 1.0}))
        by_year = {r["year"]: r for r in simulate(p, 0.0).rows}
        row = by_year[2027]
        self.assertAlmostEqual(row["tax_state"], (row["agi"] - 2_000.0) * 0.029, delta=0.05)   # exempt Social Security: none here
        self.assertGreater(by_year[2029]["irmaa_lookback_magi"], 50_000.0)
        self.assertAlmostEqual(by_year[2029]["irmaa_lookback_magi"], by_year[2027]["agi"], delta=0.01)

    def test_the_marginal_rate_includes_investment_income(self):
        p = self.base(taxable_value=3_250_000.0, taxable_basis=3_250_000.0, allocations=mix(taxable={"bonds": 1.0}))
        row = simulate(p, 0.0).rows[0]
        income = row["inv_income_ordinary"]
        want = (taxes.federal_tax(income + 1_000.0, 0.0, "mfj") - taxes.federal_tax(income, 0.0, "mfj")) / 1_000.0
        self.assertAlmostEqual(row["marginal_ordinary_rate"], want, delta=0.0006)
        self.assertGreater(row["marginal_ordinary_rate"], 0.05)

    def test_the_inherited_ira_grows_at_the_inherited_mix_return(self):
        ev = make_inh(death=date(2040, 3, 1), ira=500_000.0)
        p = self.base(tax_enabled=False, cash=10_000_000.0, inheritances=(ev,), allocations=mix(inherited={"bonds": 1.0}))
        row = {r["year"]: r for r in simulate(p, 0.0).rows}[2040]
        self.assertAlmostEqual(row["end_inh_f"], (500_000.0 - row["draw_inherited_ira"]) * 1.04, delta=0.05)

    def test_sale_proceeds_start_earning_taxable_income_in_the_year_they_arrive(self):
        ev = make_inh(ira=0.0, assets=(Asset("house", 400_000.0),))
        p = self.base(cash=1_000_000.0, inheritances=(ev,), allocations=mix(taxable={"bonds": 1.0}))
        by_year = {r["year"]: r for r in simulate(p, 0.0).rows}
        self.assertAlmostEqual(by_year[2040]["inheritance_cash_received"], 400_000.0)
        self.assertGreater(by_year[2040]["inv_income_ordinary"], by_year[2039]["inv_income_ordinary"] + 10_000.0)

    def test_the_after_tax_money_flow_balances_with_investment_income(self):
        p = self.base(cash=300_000.0, taxable_value=800_000.0, taxable_basis=500_000.0, trad1=600_000.0,
                      allocations=mix(taxable={"stocks": 0.5, "bonds": 0.5}, traditional={"stocks": 0.6, "bonds": 0.4}))
        for r in simulate(p, 4_000.0).rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"] + r["shortfall_nominal"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)


class AssetClassConfigTests(unittest.TestCase):
    CLASSES_TOML = ("[[asset_class]]\nname = \"stocks\"\ntotal_return = 0.07\nincome_yield = 0.015\nqualified_share = 0.95\n"
                    "[[asset_class]]\nname = \"cash\"\ntotal_return = 0.03\nincome_yield = 0.03\n")
    ALLOC_TOML = ("[allocation]\ncash = { cash = 1.0 }\ntaxable = { stocks = 0.7, cash = 0.3 }\n"
                  "traditional = { stocks = 1.0 }\nroth = { stocks = 1.0 }\ninherited = { stocks = 1.0 }\n")

    def load(self, extra="", model_has_return=True):
        text = ConfigLoadingTests.MINIMAL.format(claims="")
        if not model_has_return:
            text = text.replace("nominal_return = 0.05\n", "")
        text = text.replace("[spending]", extra + "\n[spending]")
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return model.load_params(path)
        finally:
            os.remove(path)

    def test_classes_and_allocation_load(self):
        p = self.load(self.CLASSES_TOML + self.ALLOC_TOML)
        self.assertEqual([c.name for c in p.asset_classes], ["stocks", "cash"])
        self.assertEqual(p.asset_classes[0], AssetClass("stocks", 0.07, 0.015, 0.95))
        self.assertEqual(dict(p.allocations)["taxable"], (("stocks", 0.7), ("cash", 0.3)))

    def test_without_them_the_single_rate_is_used(self):
        p = self.load()
        self.assertEqual((p.asset_classes, p.allocations, p.nominal_return), ((), (), 0.05))

    def test_nominal_return_is_optional_only_with_classes(self):
        self.load(self.CLASSES_TOML + self.ALLOC_TOML, model_has_return=False)         # fine
        with self.assertRaises(ValueError):
            self.load(model_has_return=False)

    def test_bad_configs_are_errors(self):
        with self.assertRaises(ValueError):
            self.load(self.CLASSES_TOML)                                                  # classes but no allocation
        with self.assertRaises(ValueError):
            self.load(self.ALLOC_TOML)                                                    # allocation but no classes
        with self.assertRaises(ValueError):
            self.load(self.CLASSES_TOML + self.ALLOC_TOML.replace("0.7, cash = 0.3", "0.6, cash = 0.3"))


# ===========================================================================
# Roth conversions: fill a tax bracket (v9)
# ===========================================================================
class RothConversionTests(unittest.TestCase):
    """Zero growth and inflation, so a $2M traditional IRA and $500k of cash give exact amounts.
    2026 joint figures: standard deduction 32,200; the 12% bracket tops out at 100,800 of taxable income."""

    def base(self, **kw):
        d = dict(person1_birth=date(1965, 1, 1), trad1=2_000_000.0, cash=500_000.0, nominal_return=0.0,
                 inflation=0.0, conv_start=2027, conv_end=2029, conv_bracket=0.12)
        d.update(kw)
        return make_params(**d)

    def rows(self, p, w=2_000.0):
        return {r["year"]: r for r in simulate(p, w).rows}

    def test_partial_or_bad_settings_are_rejected(self):
        for kw in (dict(conv_end=None), dict(conv_bracket=None), dict(conv_start=2030, conv_end=2028),
                   dict(conv_bracket=0.13)):
            with self.assertRaises(ValueError, msg=str(kw)):
                self.base(**kw)

    def test_no_settings_means_no_conversions(self):
        p = self.base(conv_start=None, conv_end=None, conv_bracket=None)
        self.assertTrue(all(r["roth_conversion"] == 0.0 for r in simulate(p, 2_000.0).rows))

    def test_the_block_is_read_from_the_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(ConfigLoadingTests.MINIMAL.format(
                claims="[roth_conversion]\nstart_year = 2030\nend_year = 2035\ntarget_bracket = 0.22"))
            path = f.name
        try:
            p = model.load_params(path)
        finally:
            os.remove(path)
        self.assertEqual((p.conv_start, p.conv_end, p.conv_bracket), (2030, 2035, 0.22))

    def test_it_fills_the_bracket_exactly_and_the_tax_comes_from_cash(self):
        row = self.rows(self.base())[2027]
        self.assertAlmostEqual(row["roth_conversion"], 100_800 + 32_200, places=2)          # 133,000
        self.assertAlmostEqual(row["tax_federal"], 0.10 * 24_800 + 0.12 * (100_800 - 24_800), places=2)  # 11,600
        self.assertAlmostEqual(row["draw_cash"], 24_000 + row["tax_federal"], places=2)    # spending + conversion tax
        self.assertEqual(row["draw_trad"], 0.0)                                              # the IRA is not tapped for the tax
        self.assertAlmostEqual(row["end_trad1"], 2_000_000 - 133_000, places=2)
        self.assertAlmostEqual(row["end_roth1"], 133_000, places=2)
        self.assertAlmostEqual(row["marginal_ordinary_rate"], 0.22, places=4)               # next dollar is in the 22% bracket

    def test_only_the_window_years_convert(self):
        rows = self.rows(self.base())
        self.assertTrue(all(rows[y]["roth_conversion"] > 0 for y in (2027, 2028, 2029)))
        self.assertTrue(all(rows[y]["roth_conversion"] == 0 for y in (2026 + 5, 2035)))

    def test_it_stops_at_what_is_left_in_the_account(self):
        row = self.rows(self.base(trad1=50_000.0))[2027]
        self.assertAlmostEqual(row["roth_conversion"], 50_000.0, places=2)
        self.assertAlmostEqual(row["end_trad1"], 0.0, places=2)

    def test_no_conversion_when_income_already_fills_the_bracket(self):
        # born 1950: a $5M IRA forces an RMD of about $218k, far above the top of the 12% bracket
        p = self.base(person1_birth=date(1950, 1, 1), trad1=5_000_000.0)
        self.assertEqual(self.rows(p)[2027]["roth_conversion"], 0.0)

    def test_taxable_social_security_made_by_the_conversion_is_counted(self):
        p = self.base(person1_birth=date(1955, 1, 1), ss_fra_monthly1=3_000.0)             # already collecting in 2027
        row = self.rows(p)[2027]
        self.assertGreater(row["ss_taxable"], 0.0)
        self.assertAlmostEqual(row["agi"] - 32_200, 100_800, delta=1.0)                      # ordinary taxable income = top of bracket
        self.assertLess(row["roth_conversion"], 133_000 - 0.5 * row["ss_gross"])             # smaller: SS took part of the room

    def test_inherited_iras_are_not_converted_but_their_income_uses_up_room(self):
        p = self.base(inheritances=(make_inh(death=date(2028, 3, 1), birth=date(1940, 1, 1), ira=500_000.0),))
        row = self.rows(p)[2028]
        self.assertAlmostEqual(row["end_inh_f"], 500_000.0 - row["draw_inherited_ira"], places=2)   # untouched by conversion
        self.assertAlmostEqual(row["roth_conversion"] + row["draw_inherited_ira"], 133_000, places=2)

    def test_conversions_go_to_the_same_persons_roth_older_person_first(self):
        p = self.base(person2_birth=date(1960, 1, 1), trad1=1_000_000.0, trad2=1_000_000.0)
        row = self.rows(p)[2027]
        self.assertAlmostEqual(row["end_roth2"], 133_000, places=2)                          # person 2 is older
        self.assertEqual(row["end_roth1"], 0.0)

    def test_conversion_income_raises_state_tax_and_the_money_flows_still_balance(self):
        p = self.base(state_origin="IN")
        for r in simulate(p, 2_000.0).rows:
            net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                   - r["penalty_early_withdrawal"] - r["surplus_to_cash"] + r["shortfall_nominal"])
            self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL)
        row = self.rows(p)[2027]
        self.assertAlmostEqual(row["tax_state"], (row["agi"] - 1_000.0) * 0.029, delta=0.05)   # agi includes the conversion


# ===========================================================================
# Cross-cutting invariants (the highest-leverage tests)
# ===========================================================================
def _scenario_configs():
    """Representative households: unlocked, locked (penalty years), and a couple."""
    common = dict(cash=200_000.0, taxable_value=300_000.0, taxable_basis=150_000.0,
                  trad1=400_000.0, roth1=100_000.0, ss_fra_monthly1=2_000.0)
    configs = {
        "unlocked": dict(common, person1_birth=date(1965, 1, 1)),
        "locked": dict(common, person1_birth=date(1975, 1, 1)),
        "couple": dict(common, person1_birth=date(1975, 1, 1), person2_birth=date(1968, 1, 1),
                       trad2=250_000.0, roth2=50_000.0, ss_fra_monthly2=1_500.0),
        # RMDs active from the first year, with a first death along the way
        "rmd_couple": dict(common, person1_birth=date(1950, 1, 1), person2_birth=date(1952, 1, 1),
                           trad2=300_000.0, roth2=50_000.0, ss_fra_monthly2=1_500.0,
                           person1_death=date(2035, 6, 1)),
        # first death early: single filing, survivor Social Security, deceased's accounts unlocked
        "couple_early_death": dict(common, person1_birth=date(1975, 1, 1), person2_birth=date(1968, 1, 1),
                                   trad2=250_000.0, roth2=50_000.0, ss_fra_monthly2=1_500.0,
                                   person1_death=date(2040, 6, 1), person2_death=date(2062, 3, 1),
                                   state_origin="IN", local_tax_rate=0.017),
        # Indiana with a county tax, then a move to a no-tax state mid-year
        "state_move": dict(common, person1_birth=date(1975, 1, 1), person2_birth=date(1968, 1, 1),
                           trad2=250_000.0, roth2=50_000.0, ss_fra_monthly2=1_500.0,
                           state_origin="IN", local_tax_rate=0.017, state_dest="TX",
                           move_date=date(2040, 3, 1)),
    }
    health_cfg = dict(health_enabled=True, health_marketplace_monthly=1_500.0,
                      health_medicare_other_monthly=200.0, health_premium_growth=0.01)
    for name in ("couple", "rmd_couple", "couple_early_death", "state_move"):
        configs[name].update(health_cfg)
    asset_cfg = dict(asset_classes=CLASSES,
                     allocations=mix(taxable={"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
                                     traditional={"stocks": 0.5, "bonds": 0.5}, inherited={"stocks": 0.4, "bonds": 0.6}))
    for name in ("unlocked", "couple", "couple_early_death"):
        configs[name].update(asset_cfg)
    # both parents die inside the horizon: inherited IRAs (even and min policies are covered in the
    # random tests), asset sales, IRMAA reacting to the extra income, and a state move on top
    inherit = (Inheritance("father", date(2034, 3, 1), birth_date=date(1945, 1, 1), ira_value=600_000.0,
                           assets=(Asset("house", 300_000.0, months_after=9, cost_pct=0.07),)),
               Inheritance("mother", date(2039, 8, 1), birth_date=date(1947, 1, 1), ira_value=400_000.0,
                           assets=(Asset("house", 250_000.0, months_after=14, cost_pct=0.07),)))
    configs["inheritance_couple"] = dict(
        common, person1_birth=date(1970, 1, 1), person2_birth=date(1972, 1, 1), trad2=250_000.0, roth2=50_000.0,
        ss_fra_monthly2=1_500.0, state_origin="IN", local_tax_rate=0.017, state_dest="TX",
        move_date=date(2045, 6, 1), inheritances=inherit, **health_cfg, **asset_cfg)
    # spousal top-up active (PIA 3,200 vs 1,100), claimed at different ages, with a death
    configs["spousal_couple"] = dict(
        common, person1_birth=date(1965, 1, 1), person2_birth=date(1967, 1, 1), ss_fra_monthly1=3_200.0,
        ss_fra_monthly2=1_100.0, ss_claim_months1=70 * 12, ss_claim_months2=63 * 12 + 4,
        person1_death=date(2050, 6, 1), **asset_cfg)
    configs["inheritance_min_policy"] = dict(configs["inheritance_couple"], inherited_ira_policy="min")
    # on Medicare from the first years with high income and a first death: IRMAA, the lookback and the
    # widow's-trap switch to single thresholds all happen inside the horizon
    configs["health_couple"] = dict(
        common, person1_birth=date(1962, 1, 1), person2_birth=date(1963, 1, 1), trad1=1_500_000.0,
        trad2=900_000.0, roth2=50_000.0, ss_fra_monthly2=1_500.0, person1_death=date(2036, 6, 1),
        health_prior_magi=300_000.0, **health_cfg)
    return configs


class InvariantTests(unittest.TestCase):
    @slow
    def test_all_strategies_satisfy_accounting_identity(self):
        """gross + SS - tax - penalty - surplus + shortfall = spending, every row,
        for every strategy x tax on/off x household. Also checks nothing goes
        negative. (Row-level tolerance is cents: rows are rounded.)"""
        for strategy in model.STRATEGIES:
            for tax_on in (True, False):
                for label, cfg in _scenario_configs().items():
                    p = make_params(strategy=strategy, tax_enabled=tax_on, **cfg)
                    for r in simulate(p, 5_000.0).rows:
                        msg = f"{strategy} tax={tax_on} {label} year={r['year']}"
                        net = (r["gross_withdrawal"] + r["ss_gross"] - r["tax_federal"] - r["tax_state"]
                               - r["penalty_early_withdrawal"] - r["surplus_to_cash"]
                               + r["shortfall_nominal"])
                        self.assertAlmostEqual(net, r["need_nominal"], delta=ROW_CENT_TOL, msg=msg)
                        for key in ("draw_cash", "draw_taxable", "draw_trad", "draw_roth",
                                    "draw_inherited_ira", "inherited_ira_required",
                                    "inheritance_ira_received", "inheritance_cash_received",
                                    "tax_federal", "tax_state", "penalty_early_withdrawal",
                                    "surplus_to_cash", "shortfall_nominal"):
                            self.assertGreaterEqual(r[key], 0.0, msg=f"{msg} {key}")
                        for key in ("end_cash", "end_taxable", "end_trad1", "end_trad2",
                                    "end_roth1", "end_roth2", "end_inh_f", "end_inh_m"):
                            self.assertGreaterEqual(r[key], -0.01, msg=f"{msg} {key}")

    def test_year_level_identity_to_float_precision(self):
        """Same identity at full float precision, by calling the per-year solver
        directly (output rows are rounded, so they cannot show 1e-6)."""
        rng = random.Random(7)
        for trial in range(80):
            state_cfg = {}
            if rng.random() < 0.5:           # Indiana with a county tax, sometimes moving to Texas
                state_cfg = dict(state_origin="IN", local_tax_rate=rng.choice([0.0, 0.017]))
                if rng.random() < 0.5:
                    state_cfg.update(state_dest="TX", move_date=date(rng.choice([2030, 2040, 2050]),
                                                                     rng.choice([1, 6, 12]), 1))
            asset_cfg = {}
            if rng.random() < 0.5:           # asset classes: yearly taxable dividends/interest, per-account returns
                w = rng.choice([0.2, 0.5, 0.8])
                asset_cfg = dict(asset_classes=CLASSES,
                                 allocations=mix(taxable={"stocks": w, "bonds": 1 - w},
                                                 traditional={"stocks": 1 - w, "bonds": w},
                                                 inherited={"bonds": 1.0}))
            inh_cfg = {}
            if rng.random() < 0.5:           # a father who dies inside the horizon, any policy, assets or not
                inh_cfg = dict(
                    inheritances=(Inheritance(
                        "father", date(rng.choice([2033, 2038, 2044]), 3, 1), birth_date=date(1940, 1, 1),
                        ira_value=rng.choice([0.0, 300_000.0, 700_000.0]), share=rng.choice([0.5, 1.0]),
                        assets=(Asset("house", rng.choice([0.0, 250_000.0]), months_after=rng.choice([0, 9, 15]),
                                      cost_pct=0.07),)),),
                    inherited_ira_policy=rng.choice(["even", "min"]))
            couple = {}
            if rng.random() < 0.5:           # couples, with an early first death in some trials
                couple = dict(person2_birth=date(1968, 1, 1), trad2=rng.uniform(0, 300_000),
                              roth2=rng.uniform(0, 200_000), ss_fra_monthly2=rng.choice([0.0, 1_500.0]),
                              person1_death=date(rng.choice([2036, 2045, 2070]), 6, 1))
            p = make_params(
                person1_birth=rng.choice([date(1950, 1, 1), date(1965, 1, 1), date(1975, 1, 1)]),
                tax_enabled=rng.choice([True, False]),
                strategy=rng.choice(list(model.STRATEGIES)),
                cash=rng.uniform(0, 300_000), taxable_value=rng.uniform(0, 400_000),
                taxable_basis=rng.uniform(0, 400_000), trad1=rng.uniform(0, 600_000),
                roth1=rng.uniform(0, 300_000), ss_fra_monthly1=rng.choice([0.0, 2_500.0]), **couple, **state_cfg, **inh_cfg, **asset_cfg)
            year = rng.choice([y for y in (2027, 2033, 2040, 2055) if y < p.last_death_year])
            py = replace(p, filing_status=p.filing_status_for(year))
            k = year - p.start_year
            price = (1 + p.inflation) ** k
            scale = (1 + p.inflation) ** (year - taxes.BASE_YEAR)
            bal = {"cash": p.cash, "taxable": p.taxable_value, "trad1": p.trad1,
                   "trad2": p.trad2, "roth1": p.roth1, "roth2": p.roth2, "inh_f": 0.0, "inh_m": 0.0}
            if inh_cfg and year >= inh_cfg["inheritances"][0].death_date.year:
                bal["inh_f"] = rng.uniform(0, 600_000)
            locked = p.locked_buckets(year)
            steps = model._year_steps(py, locked, scale)
            ss = model._ss_income(p, year, price)
            need = rng.uniform(20_000, 150_000) * price
            forced, _ = model._rmd_forced(p, year, bal)
            inh_forced, _ = model._inherited_forced(p, year, bal)
            forced = {b: forced[b] + inh_forced[b] for b in forced}
            draws, yt, pen, trad_inc, gains, surplus, short = model._solve_year(
                need, bal, p.taxable_basis, ss, scale, py, steps, locked, forced, model._state_tax_fn(p, year))
            lhs = sum(draws.values()) + ss - yt.income_tax - yt.state_tax - pen - surplus + short
            self.assertAlmostEqual(lhs, need, delta=1e-6, msg=f"trial {trial}")
            for b in forced:                          # forced draws are always part of the draws
                self.assertGreaterEqual(draws[b], forced[b] - 1e-9, msg=f"trial {trial} {b}")

    @slow
    def test_shortfall_and_final_balance_are_monotone_in_w(self):
        """The solver's bisection is only valid if more spending never helps.
        Guardrail: any strategy change that breaks this will make solve_max_w lie."""
        rng = random.Random(42)
        for trial in range(50):
            state_cfg = {}
            if rng.random() < 0.5:
                state_cfg = dict(state_origin="IN", local_tax_rate=rng.choice([0.0, 0.017]))
                if rng.random() < 0.5:
                    state_cfg.update(state_dest="TX", move_date=date(rng.choice([2030, 2040, 2050]),
                                                                     rng.choice([1, 6, 12]), 1))
            health_cfg = {}
            if rng.random() < 0.6:           # health premiums, with IRMAA driven by a prior-year MAGI
                health_cfg = dict(health_enabled=True, health_marketplace_monthly=rng.choice([0.0, 1_500.0]),
                                  health_medicare_other_monthly=rng.choice([0.0, 250.0]),
                                  health_premium_growth=rng.choice([0.0, 0.01]),
                                  health_prior_magi=rng.choice([0.0, 120_000.0, 300_000.0]))
            asset_cfg = {}
            if rng.random() < 0.5:           # asset classes: yearly taxable dividends/interest, per-account returns
                w = rng.choice([0.2, 0.5, 0.8])
                asset_cfg = dict(asset_classes=CLASSES,
                                 allocations=mix(taxable={"stocks": w, "bonds": 1 - w},
                                                 traditional={"stocks": 1 - w, "bonds": w},
                                                 inherited={"bonds": 1.0}))
            inh_cfg = {}
            if rng.random() < 0.5:           # a father who dies inside the horizon, any policy, assets or not
                inh_cfg = dict(
                    inheritances=(Inheritance(
                        "father", date(rng.choice([2033, 2038, 2044]), 3, 1), birth_date=date(1940, 1, 1),
                        ira_value=rng.choice([0.0, 300_000.0, 700_000.0]), share=rng.choice([0.5, 1.0]),
                        assets=(Asset("house", rng.choice([0.0, 250_000.0]), months_after=rng.choice([0, 9, 15]),
                                      cost_pct=0.07),)),),
                    inherited_ira_policy=rng.choice(["even", "min"]))
            couple = {}
            if rng.random() < 0.5:
                couple = dict(person2_birth=date(1968, 1, 1), trad2=rng.uniform(0, 300_000),
                              roth2=rng.uniform(0, 200_000), ss_fra_monthly2=rng.choice([0.0, 1_500.0]),
                              person1_death=date(rng.choice([2036, 2045, 2070]), 6, 1))
            p = make_params(
                person1_birth=rng.choice([date(1950, 1, 1), date(1965, 1, 1), date(1975, 1, 1)]),
                cash=rng.uniform(0, 300_000), taxable_value=rng.uniform(0, 300_000),
                taxable_basis=rng.uniform(0, 300_000), trad1=rng.uniform(0, 500_000),
                roth1=rng.uniform(0, 500_000), ss_fra_monthly1=rng.choice([0.0, 2_000.0]),
                strategy=rng.choice(list(model.STRATEGIES)), **couple, **state_cfg, **health_cfg, **inh_cfg, **asset_cfg)
            prev_short = prev_final = None
            for w in (1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0):
                res = simulate(p, w)
                if prev_short is not None:
                    self.assertGreaterEqual(res.shortfall_total + 1e-6, prev_short,
                                            msg=f"trial {trial}: shortfall fell at w={w}")
                    if res.success and prev_final is not None:
                        self.assertLessEqual(res.final_nominal, prev_final + 1e-6,
                                             msg=f"trial {trial}: final balance rose at w={w}")
                prev_short = res.shortfall_total
                prev_final = res.final_nominal if res.success else None


# ===========================================================================
# Golden regression
# ===========================================================================
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "golden", "representative_scenario.json")


def _golden_run():
    """One scenario touching: couple, SS for both, penalties, three accounts types,
    two purchases, spending curve, and the taxable_std_then_cash strategy."""
    p = make_params(
        person1_birth=date(1972, 6, 15), person2_birth=date(1974, 9, 2),
        cash=40_000.0, taxable_value=250_000.0,
        taxable_basis=150_000.0, trad1=700_000.0, trad2=500_000.0, roth1=100_000.0,
        roth2=60_000.0, ss_fra_monthly1=3_200.0, ss_fra_monthly2=1_200.0,     # spousal top-up active (1,600 > 1,200)
        strategy="taxable_std_then_cash",
        state_origin="IN", local_tax_rate=0.017, state_dest="TX", move_date=date(2045, 6, 1),
        health_enabled=True, health_marketplace_monthly=1_500.0, health_medicare_other_monthly=250.0,
        health_premium_growth=0.01,
        asset_classes=CLASSES, allocations=mix(taxable={"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
                                               traditional={"stocks": 0.6, "bonds": 0.4}, inherited={"stocks": 0.5, "bonds": 0.5}),
        inheritances=(
            Inheritance("father", date(2040, 3, 1), birth_date=date(1948, 5, 10), ira_value=400_000.0,
                        assets=(Asset("house", 250_000.0, months_after=9, cost_pct=0.07),)),
            Inheritance("mother", date(2046, 11, 15), birth_date=date(1950, 8, 20), ira_value=300_000.0,
                        assets=(Asset("house", 200_000.0, months_after=12, cost_pct=0.07),
                                Asset("belongings", 40_000.0, months_after=18, cost_pct=0.10)))),
        purchases=((2030, 40_000.0, "car"), (2040, 25_000.0, "trip")),
        curve=((60.0, 1.05), (75.0, 1.0), (85.0, 0.85)))
    res = simulate(p, 6_000.0, use_curve=True)
    return {
        "summary": {"final_nominal": res.final_nominal, "final_real": res.final_real,
                    "shortfall_total": res.shortfall_total, "lifetime_tax": res.lifetime_tax,
                    "lifetime_penalty": res.lifetime_penalty,
                    "penalty_years": res.penalty_years, "depletion_year": res.depletion_year},
        "rows": res.rows,
    }


class GoldenRegressionTests(unittest.TestCase):
    def test_representative_scenario_matches_golden_file(self):
        actual = json.loads(json.dumps(_golden_run()))       # normalize via JSON round-trip
        if os.getenv("UPDATE_GOLDEN"):
            os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
            with open(GOLDEN_PATH, "w") as f:
                json.dump(actual, f, indent=1, sort_keys=True)
            self.skipTest(f"golden file rewritten: {GOLDEN_PATH}")
        self.assertTrue(os.path.exists(GOLDEN_PATH),
                        "golden file missing; run with UPDATE_GOLDEN=1 to create it")
        with open(GOLDEN_PATH) as f:
            golden = json.load(f)
        self.assertEqual(len(actual["rows"]), len(golden["rows"]))
        for key, want in golden["summary"].items():
            got = actual["summary"][key]
            if (isinstance(want, (int, float)) and not isinstance(want, bool)
                    and isinstance(got, (int, float)) and not isinstance(got, bool)):
                self.assertAlmostEqual(got, want, delta=0.05, msg=f"summary {key}")
            else:
                self.assertEqual(got, want, msg=f"summary {key}")
        for got_row, want_row in zip(actual["rows"], golden["rows"]):
            self.assertEqual(set(got_row), set(want_row), msg="row columns changed")
            for key, want in want_row.items():
                got = got_row[key]
                if isinstance(want, (int, float)):
                    self.assertAlmostEqual(got, want, delta=0.011,
                                           msg=f"year {want_row['year']} column {key}")
                else:
                    self.assertEqual(got, want, msg=f"year {want_row['year']} column {key}")


if __name__ == "__main__":
    unittest.main()
