import unittest

import numpy as np

from FuzzyRules.mamdani import (
    FuzzyRule,
    FuzzyVariable,
    MamdaniSystem,
    TrapezoidalMF,
    TriangularMF,
)


class MembershipTests(unittest.TestCase):
    def test_triangular_anchors(self):
        mf = TriangularMF(0.0, 0.5, 1.0)
        self.assertEqual(mf(0.0), 0.0)
        self.assertEqual(mf(0.5), 1.0)
        self.assertEqual(mf(1.0), 0.0)

    def test_trapezoidal_shoulders_include_domain_edges(self):
        low = TrapezoidalMF(0.0, 0.0, 0.2, 0.45)
        high = TrapezoidalMF(0.55, 0.8, 1.0, 1.0)
        self.assertEqual(low(0.0), 1.0)
        self.assertEqual(high(1.0), 1.0)

    def test_standard_partition_has_no_hole(self):
        terms = (
            TrapezoidalMF(0.0, 0.0, 0.25, 0.50),
            TriangularMF(0.25, 0.50, 0.75),
            TrapezoidalMF(0.50, 0.75, 1.0, 1.0),
        )
        grid = np.linspace(0.0, 1.0, 1001)
        coverage = np.maximum.reduce([mf(grid) for mf in terms])
        self.assertGreater(float(np.min(coverage)), 0.0)


class MamdaniTests(unittest.TestCase):
    def test_symmetric_rule_has_symmetric_centroid(self):
        variable = FuzzyVariable(
            "x", {"middle": TriangularMF(0.0, 0.5, 1.0)}
        )
        output = FuzzyVariable(
            "y", {"middle": TriangularMF(0.0, 0.5, 1.0)}
        )
        system = MamdaniSystem(
            (variable,), output,
            (FuzzyRule(("middle",), "middle", "symmetric"),),
            output_points=401,
        )
        self.assertAlmostEqual(system.infer({"x": 0.5}), 0.5, places=12)

    def test_trace_exposes_rule_and_output_activations(self):
        variable = FuzzyVariable("x", {"low": TrapezoidalMF(0, 0, 0.4, 1)})
        output = FuzzyVariable("y", {"low": TrapezoidalMF(0, 0, 0.4, 1)})
        system = MamdaniSystem(
            (variable,), output, (FuzzyRule(("low",), "low", "r1"),)
        )
        _, trace = system.infer({"x": 0.2}, return_trace=True)
        self.assertEqual(trace["output_activation"]["low"], 1.0)
        self.assertEqual(trace["rule_strengths"], (("r1", 1.0),))


if __name__ == "__main__":
    unittest.main()
