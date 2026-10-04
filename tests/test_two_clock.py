import numpy as np
import pytest

from steercert.two_clock import (
    Lattice,
    analytic_read_auc,
    analytic_write_norm,
    counting_baseline,
    make_schedule,
    normalise_curve,
    remask_recoverable,
    shape_gap,
)


@pytest.mark.parametrize("order", ["random", "identity"])
def test_read_rises_and_write_falls_along_the_clock(order):
    schedule = make_schedule(8, 8, seed=17, order=order)
    reads = [analytic_read_auc(schedule, t, 1.0, 1.0) for t in range(schedule.length + 1)]
    writes = [analytic_write_norm(schedule, t, 2.0) for t in range(schedule.length + 1)]
    assert all(b >= a for a, b in zip(reads, reads[1:]))
    assert all(b <= a for a, b in zip(writes, writes[1:]))
    assert reads[0] == 0.5 and writes[-1] == 0.0
    if order == "random":
        # E7 schedule: decodability first peaks exactly where controllability is gone
        assert writes[int(np.argmax(reads))] == 0.0
    else:
        # left-to-right: evidence is fully read while every write is still open
        assert reads[8] == max(reads) and writes[8] == max(writes)


def test_read_and_write_position_sets_are_disjoint_at_every_step():
    schedule = make_schedule(6, 6, seed=3)
    for step in range(schedule.length + 1):
        committed_read = np.intersect1d(schedule.committed(step), schedule.read_positions)
        masked_write = np.intersect1d(schedule.masked(step), schedule.write_positions)
        assert np.intersect1d(committed_read, masked_write).size == 0


def test_remasking_recovers_only_write_positions():
    schedule = make_schedule(4, 4, seed=0, order="identity")
    # identity order commits reads first, so at step 6 two writes are committed
    assert remask_recoverable(schedule, 6, 10) == 2
    assert remask_recoverable(schedule, 4, 10) == 0


def test_shape_gap_and_counting_baseline():
    counts = np.array([8, 6, 4, 2, 1])
    baseline = counting_baseline(counts)
    assert baseline[0] == 1.0
    assert shape_gap(np.sqrt(counts) * 7.0, baseline) == pytest.approx(0.0, abs=1e-12)
    assert shape_gap(np.array([1.0, 0.2, 0.1, 0.05, 0.0]), baseline) > 0.4
    np.testing.assert_allclose(normalise_curve([0, 0]), [0, 0])
    with pytest.raises(ValueError):
        counting_baseline([-1])
    with pytest.raises(ValueError):
        shape_gap([1, 2], [1, 2, 3])


def test_lattice_falsifier_checks():
    lattice = Lattice(
        read=[[0.95, 1.0, 1.0], [0.9, 0.97, 1.0]],
        write=[[0.07, 0.04, 0.02], [0.05, 0.05, 0.06]],
        steps=[0.1, 0.5, 0.9],
        depths=(1, 2),
    )
    assert lattice.write_decreasing() == [True, False]
    assert lattice.clocks_dissociated() == [True, False]
    assert lattice.summary()["write_argmax_step"] == [0.1, 0.9]
    assert len(lattice.joint_optimum()) == 2
    with pytest.raises(ValueError):
        Lattice(read=[[1.0]], write=[[1.0, 2.0]], steps=[0.1], depths=(1,))
