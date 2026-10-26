"""Metric scorers on synthetic records (CPU, COCO ground truth stubbed)."""

import pytest

import evaluation.classifiers.chair_objects as chair_objects
from evaluation.classifiers.metrics import (
    compute_metric_records, score_chair_records, score_pope_records)


def _chair_rec(coco_id, caption):
    return {"response": caption, "metadata": {"raw": {"coco_id": coco_id}}}


@pytest.fixture
def stub_gt(monkeypatch):
    gt = {1: {"dog", "person"}, 2: {"cat", "couch", "remote"}}
    monkeypatch.setattr(chair_objects, "gt_objects_for_image",
                        lambda image_id: gt.get(int(image_id), set()))
    return gt


def test_chair_and_recall(stub_gt):
    records = [
        _chair_rec(1, "A man walks his dog past a parked car."),   # car hallucinated
        _chair_rec(2, "A cat sleeps on a couch."),                 # clean, misses remote
    ]
    s = score_chair_records(records)
    assert s["chair_i"] == pytest.approx(1 / 5)
    assert s["chair_s"] == pytest.approx(1 / 2)
    assert s["recall"] == pytest.approx(4 / 5)
    assert s["n_nonempty"] == 2


def test_chair_empty_captions_are_excluded(stub_gt):
    records = [_chair_rec(1, "A dog."), _chair_rec(2, "   ")]
    s = score_chair_records(records)
    assert s["n_empty"] == 1
    assert s["chair_s"] == 0.0
    assert s["recall"] == pytest.approx(1 / 2)


def test_pope_scoring():
    records = [
        {"ground_truth": "yes", "response": "Yes, there is.", "metadata": {"category": "random"}},
        {"ground_truth": "no", "response": "No.", "metadata": {"category": "random"}},
        {"ground_truth": "no", "response": "Yes", "metadata": {"category": "random"}},
        {"ground_truth": "yes", "response": "I cannot tell.", "metadata": {"category": "random"}},
    ]
    s = score_pope_records(records)
    assert s["accuracy_overall"] == pytest.approx(2 / 4)
    assert s["n_unparsed"] == 1
    assert s["precision_overall"] == pytest.approx(1 / 2)
    assert s["recall_overall"] == pytest.approx(1 / 2)


def test_dispatch_unknown_benchmark():
    assert compute_metric_records([], "nope")["metric"] == "unknown"
