from simpletransformers.config.model_args import ClassificationArgs
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    all_newsgroups = fetch_20newsgroups(subset="train")
    print("All 20 newsgroup names:\n", all_newsgroups.target_names)
    newsgroups_train = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    newsgroups_test = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    train_df = pd.DataFrame({"text": newsgroups_train.data, "labels": newsgroups_train.target})
    test_df = pd.DataFrame({"text": newsgroups_test.data, "labels": newsgroups_test.target})
    train_df, dev_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=0,
        stratify=train_df[["labels"]],
    )
    return train_df, dev_df, test_df


def train_bert(train_df, dev_df):
    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.output_dir = "../../Files/Outputs/topic"
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.num_train_epochs = 4
    model_args.train_batch_size = 32
    model_args.learning_rate = 4e-6
    model_args.max_seq_length = 256
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_metric_minimize = True
    model_args.early_stopping_patience = 2
    model_args.evaluate_during_training_steps = 100
    steps_per_epoch = int(np.ceil(len(train_df) / float(model_args.train_batch_size)))
    print(f"Each epoch will have {steps_per_epoch:,} steps.")
    num_labels = len(train_df["labels"].unique())
    model = ClassificationModel(
        "bert", "bert-base-cased", num_labels=num_labels, args=model_args, use_cuda=True
    )
    _, history = model.train_model(train_df, eval_df=dev_df)
    train_loss = history["train_loss"]
    eval_loss = history["eval_loss"]
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Training loss")
    plt.plot(eval_loss, label="Evaluation loss")
    plt.title("Training vs. Evaluation loss")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_classification_report(model_dir, test_df, label_names, use_cuda=False):
    args = ClassificationArgs()
    args.evaluate_during_training = False
    model = ClassificationModel(
        "bert", model_dir, num_labels=len(label_names), args=args, use_cuda=use_cuda
    )
    result, raw_outputs, _ = model.eval_model(test_df)
    y_pred = [int(np.argmax(logits)) for logits in raw_outputs]
    y_true = test_df["labels"].tolist()
    print("SimpleTransformers eval metrics:", result)
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    train_df, dev_df, topic_test_df = load_data()
    #train_bert(train_df, dev_df)
    label_names = [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]
    generate_classification_report(
        model_dir="../../Files/Outputs/topic",
        test_df=topic_test_df,
        label_names=label_names,
        use_cuda=False,
    )
