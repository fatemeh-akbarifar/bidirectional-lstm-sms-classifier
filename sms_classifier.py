"""SMS spam classification with an embedding and bidirectional LSTM."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from data_utils import download


def load_messages(path):
    frame = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    if frame.empty or frame.isna().any().any() or not frame.label.isin(["ham", "spam"]).all():
        raise ValueError("Expected nonempty TSV rows containing ham/spam and a message")
    frame["label"] = frame.label.map({"ham": 0, "spam": 1}).astype("float32")
    return frame


def split_messages(pool, test, seed=42):
    overlap = len(set(pool.message) & set(test.message))
    pool = pool.loc[~pool.message.isin(test.message)].drop_duplicates("message")
    train, validation = train_test_split(pool, test_size=.2, random_state=seed, stratify=pool.label)
    return train, validation, overlap


def build_model(training_messages, sequence_length=120):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=12000, output_sequence_length=sequence_length)
    vectorizer.adapt(tf.constant(list(training_messages)))
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 32, mask_zero=True)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=.2))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def predict_message(model, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide a nonempty message")
    probability = float(model(tf.constant([text]), training=False).numpy()[0, 0])
    return [probability, "spam" if probability >= .5 else "ham"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, help="Load a saved .keras model for inference")
    parser.add_argument("--message", help="Message to classify")
    args = parser.parse_args()
    if args.model:
        if not args.message:
            parser.error("--model requires --message")
        model = tf.keras.models.load_model(args.model)
        print(json.dumps(predict_message(model, args.message)))
        return
    if args.epochs < 1:
        parser.error("epochs must be positive")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    train_path = download("https://cdn.freecodecamp.org/project-data/sms/train-data.tsv", args.data_dir / "train-data.tsv")
    test_path = download("https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv", args.data_dir / "valid-data.tsv")
    pool, test = load_messages(train_path), load_messages(test_path)
    train, validation, overlap = split_messages(pool, test, args.seed)
    model = build_model(train.message)
    history = model.fit(tf.constant(train.message.tolist()), train.label.to_numpy(),
                        validation_data=(tf.constant(validation.message.tolist()), validation.label.to_numpy()),
                        epochs=args.epochs, batch_size=32, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    probabilities = model.predict(tf.constant(test.message.tolist()), verbose=0).ravel()
    predicted = (probabilities >= .5).astype(int)
    metrics = {"classification_report": classification_report(test.label, predicted, labels=[0, 1], target_names=["ham", "spam"], output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(test.label, predicted, labels=[0, 1]).tolist(),
               "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
               "epochs_run": len(history.history["loss"]), "seed": args.seed,
               "source_train_test_exact_message_overlap": overlap,
               "remaining_train_test_exact_message_overlap": len((set(train.message) | set(validation.message)) & set(test.message))}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.keras")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2))
    pd.DataFrame({"actual": test.label.astype(int), "predicted": predicted, "spam_probability": probabilities}).to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    if args.message:
        print(json.dumps(predict_message(model, args.message)))


if __name__ == "__main__":
    main()
