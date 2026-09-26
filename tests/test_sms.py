import numpy as np
import pytest
import tensorflow as tf
from sms_classifier import build_model, predict_message, load_messages


def test_saved_model_accepts_raw_and_unknown_words(tmp_path):
    model = build_model(["hello friend", "win a prize", "see you soon"], sequence_length=8)
    batch = tf.constant(["hello friend", "win a prize"])
    assert np.isfinite(model.train_on_batch(batch, np.array([0.,1.]))).all()
    before = predict_message(model, "unseen vocabulary")
    path = tmp_path / "model.keras"; model.save(path)
    loaded = tf.keras.models.load_model(path)
    after = predict_message(loaded, "unseen vocabulary")
    assert 0 <= after[0] <= 1
    assert after[1] in ("ham", "spam")
    np.testing.assert_allclose(before[0], after[0], atol=1e-6)
    with pytest.raises(ValueError): predict_message(loaded, " ")


def test_invalid_labels(tmp_path):
    path = tmp_path / "input.tsv"; path.write_text("unknown\thello\n")
    with pytest.raises(ValueError): load_messages(path)


def test_message_splits_are_disjoint():
    import pandas as pd
    from sms_classifier import split_messages
    pool = pd.DataFrame({"message": [f"message {i}" for i in range(20)] * 2,
                         "label": [i % 2 for i in range(20)] * 2})
    test = pd.DataFrame({"message": ["message 0", "message 1"], "label": [0, 1]})
    train, validation, overlap = split_messages(pool, test)
    assert overlap == 2
    assert not set(train.message) & set(validation.message)
    assert not (set(train.message) | set(validation.message)) & set(test.message)
    assert len(train) + len(validation) == 18
