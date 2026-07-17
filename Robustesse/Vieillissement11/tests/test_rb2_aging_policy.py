from Common.rb2_aging_policy import make_aging_rb2_policy
from Common.rb2_policy import make_rb2_policy


def _args(soc=0.9, net=5000.0):
    return (
        soc, net, [], [], 0.0, 0.0, 1.0, 100.0, 200.0,
        10_000.0, 10_000.0, float("inf"), float("inf"), 1.0, 1.0,
    )


def test_aging_policy_is_exact_rb2_when_all_layers_are_disabled():
    rule = make_aging_rb2_policy(
        fc_min_on_h=0.0, ely_min_on_h=0.0,
        fc_reversible_trigger_uv=float("inf"),
        ely_reversible_trigger_uv=float("inf"),
        permanent_strength_fc=0.0, permanent_strength_ely=0.0,
    )
    action, _ = rule(*_args(), aging_context={})
    reference, _ = make_rb2_policy(0.59, 0.49)(*_args())
    assert action == reference


def test_high_reversible_fc_loss_requests_rest_when_soc_allows_it():
    rule = make_aging_rb2_policy(fc_reversible_trigger_uv=100.0)
    action, _ = rule(
        *_args(soc=0.9),
        aging_context={"fc": {"reversible_uv": 101.0}, "ely": {}},
    )
    assert action[1] == 0.0


def test_recovery_is_not_forced_at_low_soc():
    rule = make_aging_rb2_policy(fc_reversible_trigger_uv=100.0)
    action, _ = rule(
        *_args(soc=0.4),
        aging_context={"fc": {"reversible_uv": 101.0}, "ely": {}},
    )
    assert action[1] > 0.0
