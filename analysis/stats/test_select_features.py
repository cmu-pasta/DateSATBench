import numpy as np
import pandas as pd

from analysis.stats.select_features import medoid, most_correlated_pair, select_features


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"id": [f"i{k}" for k in range(n)], "corpus": "llm", **cols})


def test_monotone_copies_collapse_to_the_first_and_constants_are_dropped():
    a = np.arange(20.0)
    u = (a * 7) % 20                       # a permutation of a, barely correlated with it
    df = frame(a=a, b=2 * a, c=a ** 2, k=np.ones(20), u=u)
    selected, families, constant = select_features(df, 0.8)
    assert selected == ["a", "u"]
    assert families == {"a": ["b", "c"]}   # all three tie at rho 1, so the first is kept
    assert constant == ["k"]


def test_the_medoid_is_the_member_closest_to_the_rest():
    rho = pd.DataFrame([[1.0, 0.95, 0.85], [0.95, 1.0, 0.95], [0.85, 0.95, 1.0]],
                       index=list("xyz"), columns=list("xyz"))
    assert medoid(["x", "y", "z"], rho) == "y"
    assert medoid(["x", "z"], rho) == "x"  # a tie goes to the first


def test_no_two_selected_features_reach_the_threshold_and_every_feature_is_accounted_for():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 4))
    cols = {f"f{i}": base[:, i % 4] + rng.normal(scale=0.2 * (i // 4 + 1), size=200)
            for i in range(16)}
    df = frame(**cols)
    for threshold in (0.9, 0.7, 0.5):
        selected, families, constant = select_features(df, threshold)
        a, b, r = most_correlated_pair(df, selected)
        assert abs(r) < threshold
        dropped = [f for d in families.values() for f in d]
        assert sorted(selected + dropped + constant) == sorted(cols)
        assert set(families) <= set(selected)


if __name__ == "__main__":
    test_monotone_copies_collapse_to_the_first_and_constants_are_dropped()
    test_the_medoid_is_the_member_closest_to_the_rest()
    test_no_two_selected_features_reach_the_threshold_and_every_feature_is_accounted_for()
    print("all select_features tests passed")
