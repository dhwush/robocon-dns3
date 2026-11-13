from vote import LOCK_FRAMES, WINDOW_FRAMES, Vote


def test_unknown_gaps_allow_lock_within_window():
    vote = Vote()
    for _ in range(5):
        for value in ('1', '1', None, '1'):
            vote.see(value)
    assert vote.locked == '1'


def test_sparse_evidence_expires():
    vote = Vote()
    for _ in range(30):
        vote.see('1')
        vote.see(None)
    assert vote.locked is None
    assert vote.count == WINDOW_FRAMES // 2
    for _ in range(WINDOW_FRAMES):
        vote.see(None)
    assert vote.count == 0
    assert vote.pending is None


def test_conflict_restarts_evidence():
    vote = Vote()
    for _ in range(LOCK_FRAMES - 1):
        vote.see('1')
    vote.see('2')
    assert vote.pending == '2'
    assert vote.count == 1
    vote.see('1')
    assert vote.count == 1
    assert vote.locked is None


def test_move_clears_window():
    vote = Vote()
    for _ in range(LOCK_FRAMES - 1):
        vote.see('1')
    vote.on_move()
    vote.see('1')
    assert vote.count == 1
    assert vote.locked is None


def test_recheck_requires_arrival_then_tolerates_unknown():
    vote = Vote(locked='1')
    for _ in range(LOCK_FRAMES):
        vote.see('2')
    assert vote.recheck
    for _ in range(WINDOW_FRAMES):
        vote.see('2')
    assert vote.locked == '1'
    vote.ready = True
    for _ in range(5):
        for value in ('2', None, '2', '2'):
            vote.see(value)
    assert vote.locked == '2'
    assert not vote.recheck
