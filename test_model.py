"""
Sanity tests for v0. Run from this folder:  python -m unittest -v
"""
import unittest
from dataclasses import replace
from datetime import date

import taxes
from model import Params, simulate, solve_max_w


def make_params(**kw):
    base = Params(
        start_year=2027, person1_birth=date(1965, 1, 1), person2_birth=None,
        horizon_age=95, horizon_person="person1", filing_status="mfj",
        nominal_return=0.05, inflation=0.025, tax_enabled=True,
        withdrawal_order=("cash", "taxable", "trad", "roth"),
        cash=0.0, taxable_value=0.0, taxable_basis=0.0, traditional=0.0, roth=0.0,
        target_monthly_real=0.0, curve=((0.0, 1.0),), purchases=(),
    )
    return replace(base, **kw)


class TaxTests(unittest.TestCase):
    def test_ordinary_bracket_math(self):
        # MFJ taxable income 100,000 -> 10% of 24,800 + 12% of the rest
        tax = taxes.federal_tax(32_200 + 100_000, 0, "mfj")
        self.assertAlmostEqual(tax, 0.10 * 24_800 + 0.12 * (100_000 - 24_800), places=4)

    def test_zero_tax_under_standard_deduction(self):
        self.assertEqual(taxes.federal_tax(30_000, 0, "mfj"), 0.0)

    def test_gains_in_zero_bracket(self):
        self.assertEqual(taxes.federal_tax(0, 100_000, "mfj"), 0.0)

    def test_gains_stack_on_ordinary(self):
        # ordinary fills up to the 0% LTCG ceiling; extra gains taxed at 15%
        ordinary = 32_200 + 98_900
        self.assertAlmostEqual(
            taxes.federal_tax(ordinary, 10_000, "mfj"),
            taxes.federal_tax(ordinary, 0, "mfj") + 1_500.0, places=4)

    def test_indexing_scales_thresholds(self):
        self.assertAlmostEqual(taxes.federal_tax(2 * 32_200 + 2 * 50_000, 0, "mfj", scale=2.0),
                               2 * taxes.federal_tax(32_200 + 50_000, 0, "mfj"), places=4)


class EngineTests(unittest.TestCase):
    def test_zero_everything_gives_simple_division(self):
        p = make_params(nominal_return=0.0, inflation=0.0, tax_enabled=False,
                        traditional=1_200_000.0)
        w = solve_max_w(p)
        self.assertAlmostEqual(w, 1_200_000.0 / (12 * p.n_years), delta=0.05)

    def test_all_roth_matches_annuity_formula(self):
        p = make_params(roth=1_000_000.0)
        rr = (1 + p.nominal_return) / (1 + p.inflation) - 1
        n = p.n_years
        annual = 1_000_000.0 / sum((1 + rr) ** (-k) for k in range(n))
        self.assertAlmostEqual(solve_max_w(p), annual / 12, delta=0.05)

    def test_after_tax_spending_equals_need(self):
        p = make_params(traditional=2_000_000.0)
        res = simulate(p, 6_000.0)
        self.assertTrue(res.success)
        for r in res.rows:
            self.assertAlmostEqual(r["gross_withdrawal"] - r["tax_federal"],
                                   r["spend_nominal"], delta=0.05)

    def test_solved_w_is_the_edge(self):
        p = make_params(traditional=1_500_000.0, roth=200_000.0,
                        taxable_value=300_000.0, taxable_basis=200_000.0, cash=50_000.0)
        w = solve_max_w(p)
        self.assertTrue(simulate(p, w).success)
        self.assertFalse(simulate(p, w + 5.0).success)
        self.assertLess(simulate(p, w).final_nominal, 5_000.0)  # ~zero left

    def test_taxes_reduce_w(self):
        p_tax = make_params(traditional=1_500_000.0)
        p_free = replace(p_tax, tax_enabled=False)
        self.assertLess(solve_max_w(p_tax), solve_max_w(p_free))

    def test_roth_first_order_matters(self):
        base = make_params(traditional=1_000_000.0, roth=1_000_000.0)
        trad_first = replace(base, withdrawal_order=("trad", "roth"))
        roth_first = replace(base, withdrawal_order=("roth", "trad"))
        # Both are valid; just confirm the solver runs and results differ.
        self.assertNotAlmostEqual(solve_max_w(trad_first), solve_max_w(roth_first), delta=1.0)

    def test_purchases_reduce_w(self):
        p = make_params(roth=1_000_000.0, purchases=((2030, 40_000.0, "car"),))
        self.assertLess(solve_max_w(p), solve_max_w(p, include_purchases=False))

    def test_cost_basis_reduces_tax(self):
        # Large enough that gains exceed the 0% LTCG bracket
        hi_basis = make_params(taxable_value=6_000_000.0, taxable_basis=6_000_000.0)
        lo_basis = replace(hi_basis, taxable_basis=1_000_000.0)
        self.assertGreater(solve_max_w(hi_basis), solve_max_w(lo_basis))


if __name__ == "__main__":
    unittest.main()
