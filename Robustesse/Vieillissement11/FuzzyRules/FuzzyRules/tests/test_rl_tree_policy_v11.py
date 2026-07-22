"""Tests unitaires de la policy 'regles apprises' par arbre (I0/IS), V11-p=2.

On n'utilise PAS le cache enseignant (218 999 pas, hors depot) : les invariants
du wrapper de policy sont isoles avec des arbres JOUETS ajustes sur des donnees
synthetiques dont on controle exactement les predictions. Ces tests verifient les
invariants obligatoires du protocole (PLAN_FUZZY_RULE_LEARNING_V11_P2 section 6) :
bilan de puissance ferme, gardes de defaillance, zone morte, ``reset()`` exact,
test nul de permutation d'une feature non utilisee, invariants de signe.
"""

import unittest

import numpy as np

from Common import Init_EMR_MG_v16_python as I
from FuzzyRules.rl_dataset_v11 import I0_FEATURES, IS_FEATURES
from FuzzyRules.rl_tree_policy_v11 import (
    POLICY_ID_STEM,
    _feature_vector,
    fit_tree,
    make_tree_policy_v11,
    null_test_unused_permutation,
)


def _toy_dataset(rows, information_set="I0", n_features=3, reps=60):
    """Construit un dataset synthetique au format attendu par ``fit_tree``.

    ``rows`` est une liste de ``(features_tuple, y)``. Chaque ligne est repliquee
    ``reps`` fois pour laisser des feuilles pures ; toutes les lignes sont dans le
    bloc 'train'. La cible ``y`` etant constante pour un vecteur de features
    donne, l'arbre predit exactement les valeurs choisies.
    """
    X, y = [], []
    for feats, target in rows:
        for _ in range(reps):
            X.append(list(feats))
            y.append(target)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    key = {"I0": "X_i0", "IS": "X_is", "IF": "X_if"}[information_set]
    return {
        key: X,
        "y": y,
        "split": np.array(["train"] * len(y), dtype="<U6"),
        "inv_freq_weight": np.ones(len(y)),
    }


def _args(net, soc=0.60, h2=100.0, soh_bat=1.0, soh_fc=1.0, soh_ely=1.0,
          failures=None, p_fc_max=None, p_ely_max=None):
    """Arguments de boucle, meme convention que les tests FLC."""
    return (
        soc, net, list(failures or []), np.zeros(1), 0.0, 0.0, soh_bat,
        h2, 200.0,
        I.FC["P_fc_max"] if p_fc_max is None else p_fc_max,
        I.ELY["P_ely_max"] if p_ely_max is None else p_ely_max,
        float("inf"), float("inf"), soh_fc, soh_ely,
    )


# Trois regimes distincts en P_net (colonne 0), SoC/H2 constants : la cible ne
# depend que de P_net, donc l'arbre ne coupe que sur P_net (SoC/H2 non utilises).
_I0_ROWS = [
    ((2000.0, 0.5, 0.5), 1000.0),   # deficit franc -> FC +1000 W
    ((-5000.0, 0.5, 0.5), -3000.0),  # surplus franc -> ELY -3000 W
    ((150.0, 0.5, 0.5), 150.0),      # micro-deficit -> pour la zone morte
]


class TreePolicyInvariantTests(unittest.TestCase):
    def setUp(self):
        ds = _toy_dataset(_I0_ROWS, "I0")
        self.tree = fit_tree(ds, "I0", max_depth=3, min_samples_leaf=1)

    # -- fidelite de l'arbre jouet (predictions controlees) -------------------
    def test_toy_tree_predicts_chosen_values(self):
        pred = self.tree.predict(np.array(
            [[2000.0, 0.5, 0.5], [-5000.0, 0.5, 0.5], [150.0, 0.5, 0.5]]
        ))
        np.testing.assert_allclose(pred, [1000.0, -3000.0, 150.0])

    # -- invariants de signe et fermeture du bilan ----------------------------
    def test_deficit_uses_fc_and_battery_only_and_closes_balance(self):
        policy = make_tree_policy_v11(self.tree)
        action, lol = policy(*_args(net=2000.0))
        self.assertGreaterEqual(action[1], 0.0)   # p_fc >= 0
        self.assertEqual(action[2], 0.0)          # p_ely == 0
        self.assertAlmostEqual(sum(action), 2000.0, places=9)
        self.assertEqual(lol, 0.0)

    def test_surplus_uses_ely_and_battery_only_and_closes_balance(self):
        policy = make_tree_policy_v11(self.tree)
        action, lol = policy(*_args(net=-5000.0))
        self.assertEqual(action[1], 0.0)          # p_fc == 0
        self.assertLessEqual(action[2], 0.0)      # p_ely <= 0
        self.assertAlmostEqual(sum(action), -5000.0, places=9)
        self.assertEqual(lol, 0.0)

    # -- zone morte -----------------------------------------------------------
    def test_deadband_zeroes_micro_command_to_battery(self):
        policy = make_tree_policy_v11(self.tree, deadband_w=200.0)
        action, _ = policy(*_args(net=150.0))
        self.assertEqual(action[1], 0.0)          # FC coupee
        self.assertEqual(action[2], 0.0)          # ELY coupee
        self.assertAlmostEqual(action[0], 150.0, places=9)  # tout batterie

    def test_deadband_zero_keeps_micro_command(self):
        policy = make_tree_policy_v11(self.tree, deadband_w=0.0)
        action, _ = policy(*_args(net=150.0))
        self.assertGreater(action[1], 0.0)        # FC non coupee

    # -- gardes de defaillance identiques a la FLC ----------------------------
    def test_fc_failure_clamps_command_to_nonpositive(self):
        policy = make_tree_policy_v11(self.tree)
        action, _ = policy(*_args(net=2000.0, failures=["FC"]))
        self.assertEqual(action[1], 0.0)          # pas de FC quand FC en panne
        self.assertAlmostEqual(sum(action), 2000.0, places=9)

    def test_ely_failure_clamps_command_to_nonnegative(self):
        policy = make_tree_policy_v11(self.tree)
        action, _ = policy(*_args(net=-5000.0, failures=["ELY"]))
        self.assertEqual(action[2], 0.0)          # pas d'ELY quand ELY en panne
        self.assertAlmostEqual(sum(action), -5000.0, places=9)

    # -- physique deleguee a la couche de securite commune --------------------
    def test_physical_power_limit_is_applied_by_get_lol(self):
        policy = make_tree_policy_v11(self.tree)
        action, _ = policy(*_args(net=2000.0, soc=0.3, p_fc_max=100.0))
        self.assertLessEqual(action[1] / I.CONV["eta"], 100.0 + 1e-6)

    # -- reset exact (policy sans memoire) ------------------------------------
    def test_reset_is_exact_for_stateless_policy(self):
        policy = make_tree_policy_v11(self.tree)
        before = policy(*_args(net=2000.0))
        policy.reset()
        after = policy(*_args(net=2000.0))
        self.assertEqual(before, after)

    # -- test nul : permuter une feature non utilisee ne change rien ----------
    def test_null_permutation_of_unused_feature_leaves_prediction_intact(self):
        feature, delta = null_test_unused_permutation(self.tree, _toy_dataset(
            _I0_ROWS, "I0"))
        self.assertIn(feature, ("SoC_norm", "E_h2_norm"))
        self.assertLess(delta, 1e-9)

    # -- identite et metadonnees explicites -----------------------------------
    def test_identity_and_metadata_are_explicit(self):
        policy = make_tree_policy_v11(self.tree)
        self.assertEqual(policy.information_set, "I0")
        self.assertTrue(policy.policy_id.startswith(f"{POLICY_ID_STEM}-i0-"))
        self.assertEqual(len(policy.rl_metadata["spec_sha256"]), 64)
        self.assertEqual(policy.rl_metadata["features"], list(I0_FEATURES))
        self.assertIn("signed", policy.rl_metadata["target"])

    def test_distinct_deadband_yields_distinct_policy_id(self):
        a = make_tree_policy_v11(self.tree, deadband_w=0.0)
        b = make_tree_policy_v11(self.tree, deadband_w=50.0)
        self.assertNotEqual(a.policy_id, b.policy_id)


class FeatureVectorTests(unittest.TestCase):
    def test_i0_shape_and_normalization_clamps(self):
        # SoC sous la borne basse -> 0 ; H2 au-dessus du reservoir -> 1.
        x = _feature_vector(1000.0, SoC_t=0.10, E_h2_t=250.0, E_h2_init=200.0,
                            information_set="I0")
        self.assertEqual(x.shape, (1, 3))
        self.assertEqual(x[0, 0], 1000.0)
        self.assertEqual(x[0, 1], 0.0)   # SoC clampe en bas
        self.assertEqual(x[0, 2], 1.0)   # H2 clampe en haut

    def test_is_appends_three_wear_columns(self):
        x = _feature_vector(1000.0, SoC_t=0.60, E_h2_t=100.0, E_h2_init=200.0,
                            information_set="IS", wear=(0.5, 0.2, 0.9))
        self.assertEqual(x.shape, (1, 6))
        np.testing.assert_allclose(x[0, 3:], [0.5, 0.2, 0.9])


class ISPolicyTests(unittest.TestCase):
    """IS : au BoL (SoH=1) l'usure est nulle ; le bilan reste ferme."""

    def setUp(self):
        # Cible ne depend que de P_net : usure non utilisee (comme un neuf).
        rows = [
            ((2000.0, 0.5, 0.5, 0.0, 0.0, 0.0), 1200.0),
            ((-5000.0, 0.5, 0.5, 0.0, 0.0, 0.0), -2500.0),
        ]
        ds = _toy_dataset(rows, "IS")
        self.tree = fit_tree(ds, "IS", max_depth=3, min_samples_leaf=1)

    def test_information_set_and_features(self):
        policy = make_tree_policy_v11(self.tree)
        self.assertEqual(policy.information_set, "IS")
        self.assertEqual(policy.rl_metadata["features"], list(IS_FEATURES))

    def test_bol_feeds_zero_wear_and_matches_direct_prediction(self):
        policy = make_tree_policy_v11(self.tree)
        action, _ = policy(*_args(net=2000.0, soh_bat=1.0, soh_fc=1.0,
                                  soh_ely=1.0))
        # Au BoL, l'usure est (0,0,0) : la prediction egale l'arbre sur ce point.
        soc_n = (0.60 - 0.20) / (0.995 - 0.20)
        h2_n = 100.0 / 200.0
        direct = float(self.tree.predict(
            np.array([[2000.0, soc_n, h2_n, 0.0, 0.0, 0.0]]))[0])
        self.assertGreaterEqual(action[1], 0.0)
        self.assertAlmostEqual(action[1], max(direct, 0.0), places=6)
        self.assertAlmostEqual(sum(action), 2000.0, places=9)


if __name__ == "__main__":
    unittest.main()
