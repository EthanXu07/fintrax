"""Fine-tune FinBERT into a 3-way confidence classifier (low / neutral / high)."""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fintrax import lexicon


class SentenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Cross-entropy weighted by inverse class frequency."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def metrics(pred) -> dict:
    y_pred = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "macro_f1": f1_score(pred.label_ids, y_pred, average="macro")}


def evaluate(trainer: Trainer, ds: Dataset, y: np.ndarray) -> dict:
    y_pred = trainer.predict(ds).predictions.argmax(-1)
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "macro_f1": round(f1_score(y, y_pred, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1, 2]).tolist(),
        "report": classification_report(y, y_pred, labels=[0, 1, 2], target_names=lexicon.LABELS,
                                        output_dict=True, zero_division=0),
    }


def run(cfg: dict) -> dict:
    tc, paths = cfg["train"], cfg["paths"]
    set_seed(cfg["label"]["seed"])
    split = {n: pd.read_parquet(paths["processed"] / f"{n}.parquet") for n in ("train", "val", "test")}

    tok = AutoTokenizer.from_pretrained(tc["base_model"])
    id2label = dict(enumerate(lexicon.LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        tc["base_model"], num_labels=3, id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True,
    )
    # FinBERT's head predicts sentiment; start the confidence head from scratch.
    model.classifier.reset_parameters()

    ds = {n: SentenceDataset(df["sentence"].tolist(), df["label"].tolist(), tok, tc["max_len"])
          for n, df in split.items()}
    counts = np.bincount(split["train"]["label"], minlength=3)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float)

    args = TrainingArguments(
        output_dir=str(paths["model"] / "checkpoints"),
        learning_rate=tc["lr"],
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=0.1,  # <1 is a ratio of total steps in transformers v5
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,  # unsupported on MPS
        seed=cfg["label"]["seed"],
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                              processing_class=tok, compute_metrics=metrics, class_weights=weights)
    trainer.train()
    trainer.save_model(str(paths["model"]))
    tok.save_pretrained(str(paths["model"]))

    test = split["test"]
    y = test["label"].to_numpy()
    # Leakage check: hide every lexicon cue. Accuracy above the majority baseline
    # means the model learned context beyond the words that generated its labels.
    masked = [lexicon.mask_cues(s, tok.mask_token) for s in test["sentence"]]
    result = {
        "train_size": len(split["train"]), "val_size": len(split["val"]), "test_size": len(test),
        "test_calls": int(test["call_id"].nunique()),
        "majority_baseline_accuracy": round(float(np.bincount(y).max() / len(y)), 4),
        "test": evaluate(trainer, ds["test"], y),
        "test_masked_cues": evaluate(trainer, SentenceDataset(masked, y.tolist(), tok, tc["max_len"]), y),
    }
    out = paths["results"] / "metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"test acc {result['test']['accuracy']} macro-F1 {result['test']['macro_f1']} | "
          f"masked acc {result['test_masked_cues']['accuracy']} | "
          f"majority {result['majority_baseline_accuracy']} -> {out}")
    return result
