from compare_surveys import frontier


def test_frontier_keeps_distance_observation_tradeoffs():
    short = dict(distance_m=1., observation_count=4)
    fewer = dict(distance_m=2., observation_count=2)
    worse = dict(distance_m=3., observation_count=4)
    assert frontier([short, fewer, worse]) == [short, fewer]



def test_frontier_does_not_compare_different_observation_units():
    old = dict(distance_m=1., observation_count=1)
    joint = dict(distance_m=2., observation_count=2, observation_unit='settled_pose')
    assert frontier([old, joint]) == [old, joint]
