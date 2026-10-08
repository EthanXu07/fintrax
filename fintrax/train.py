"""Fine-tune FinBERT into a 3-way confidence classifier (low / neutral / high)."""
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fintrax import lexicon
from fintrax.score import Scorer


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


def evaluate(scorer, df: pd.DataFrame) -> dict:
    """Metrics through the same inference path score.py uses.

    (Trainer.predict isn't used: with length-grouped sampling its predictions can
    come back in a different order than the labels.)
    """
    probs = scorer(df["sentence"].tolist())
    to_label = np.array([lexicon.LABELS.index(scorer.id2name[i]) for i in range(probs.shape[1])])
    y, y_pred = df["label"].to_numpy(), to_label[probs.argmax(1)]
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

    ds = {"train": SentenceDataset(split["train"]["sentence"].tolist(), split["train"]["label"].tolist(),
                                   tok, tc["max_len"])}
    counts = np.bincount(split["train"]["label"], minlength=3)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float)

    args = TrainingArguments(
        output_dir=str(paths["model"] / "checkpoints"),
        learning_rate=tc["lr"],
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        bf16=tc.get("bf16", False),
        train_sampling_strategy="group_by_length",  # less padding per batch
        weight_decay=0.01,
        warmup_steps=0.1,  # <1 is a ratio of total steps in transformers v5
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        dataloader_pin_memory=False,  # unsupported on MPS
        seed=cfg["label"]["seed"],
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=ds["train"], processing_class=tok,
                              class_weights=weights)
    trainer.train()
    trainer.save_model(str(paths["model"]))
    tok.save_pretrained(str(paths["model"]))

    test = split["test"]
    y = test["label"].to_numpy()
    scorer = Scorer(str(paths["model"]), tc["max_len"])
    result = {
        "train_size": len(split["train"]), "val_size": len(split["val"]), "test_size": len(test),
        "train_calls": int(split["train"]["call_id"].nunique()), "test_calls": int(test["call_id"].nunique()),
        "majority_baseline_accuracy": round(float(np.bincount(y).max() / len(y)), 4),
        "val": {k: v for k, v in evaluate(scorer, split["val"]).items() if k in ("accuracy", "macro_f1")},
        "test": evaluate(scorer, test),
        "context_check": context_check(scorer, test),
    }
    out = paths["results"] / "metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"test acc {result['test']['accuracy']} macro-F1 {result['test']['macro_f1']} | "
          f"cue-deleted low-vs-high AUC {result['context_check']['auc']} | "
          f"majority {result['majority_baseline_accuracy']} -> {out}")
    return result


def context_check(scorer, test: pd.DataFrame) -> dict:
    """Does the model know anything the lexicon doesn't?

    Test accuracy only shows the model reproduces its weak labels. Here every
    lexicon cue is deleted from the test's low/high sentences and we ask whether
    P(high) - P(low) still ranks them correctly (AUC; 0.5 = no signal beyond
    the lexicon). Neutral sentences are excluded: they have no cues to delete,
    so they would pass trivially.
    """
    cued = test[test["weak_label"] != "neutral"]
    stripped = [re.sub(r"\s+", " ", lexicon.mask_cues(s, "")).strip() for s in cued["sentence"]]
    probs, ids = scorer(stripped), scorer.label2id
    is_high = (cued["weak_label"] == "high").to_numpy()
    auc = roc_auc_score(is_high, probs[:, ids["high"]] - probs[:, ids["low"]])
    return {"n": len(cued), "auc": round(float(auc), 4),
            "share_predicted_neutral": round(float((probs.argmax(1) == ids["neutral"]).mean()), 4)}
