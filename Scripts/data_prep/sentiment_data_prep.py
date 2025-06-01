import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report
from datasets import load_dataset

nltk.download("vader_lexicon")
geo = load_dataset("go_emotions", "simplified")
POS_TAGS = {
    "admiration", "amusement", "approval", "caring", "confidence", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "relief", "surprise"
}
NEG_TAGS = {
    "anger", "annoyance", "disappointment", "disgust", "embarrassment",
    "fear", "grief", "nervousness", "remorse", "sadness"
}

def collapse_labels(example):
    tags = example["labels"]
    names = [geo["train"].features["labels"].feature.int2str(i) for i in tags]
    if any(t in POS_TAGS for t in names) and not any(t in NEG_TAGS for t in names):
        new_label = 2
    elif any(t in NEG_TAGS for t in names) and not any(t in POS_TAGS for t in names):
        new_label = 0
    else:
        new_label = 1
    return {"text": example["text"], "label": new_label}

def run_vader(sentences):
    sia = SentimentIntensityAnalyzer()
    preds = []
    for sent in sentences:
        scores = sia.polarity_scores(sent)
        comp = scores["compound"]
        if comp >= 0.05:
            preds.append(2)
        elif comp <= -0.05:
            preds.append(0)
        else:
            preds.append(1)
    return preds

def main():
    train_geo = geo["train"].map(collapse_labels, remove_columns=["labels", "id"])
    val_geo = geo["validation"].map(collapse_labels, remove_columns=["labels", "id"])
    train_df = pd.DataFrame(train_geo)
    val_df = pd.DataFrame(val_geo)

    args = ClassificationArgs()
    args.num_train_epochs = 3
    args.train_batch_size = 16
    args.max_seq_length = 128
    args.evaluate_during_training = True
    args.labels_list = [0, 1, 2]
    args.output_dir = "../../Files/Outputs/bert-sentiment"

    model = ClassificationModel(
        "bert",
        "bert-base-cased",
        num_labels=3,
        args=args,
        use_cuda=True
    )

    model.train_model(train_df.rename(columns={"label": "labels"}), eval_df=val_df.rename(columns={"label": "labels"}))

    test_path = "../../Files/sentiment-topic-test.tsv"
    df_test = pd.read_csv(test_path, sep="\t", encoding="utf-8")
    sent_test = df_test.rename(columns={"sentence": "text", "sentiment": "labels"})
    sent_test = sent_test[["text", "labels"]]
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    sent_test["labels"] = sent_test["labels"].map(mapping)

    sentences = sent_test["text"].tolist()
    y_true = sent_test["labels"].tolist()
    vader_preds = run_vader(sentences)

    result, raw_outputs, wrong_predictions = model.eval_model(sent_test)
    bert_preds = [int(np.argmax(scores)) for scores in raw_outputs]

    class_names = ["negative", "neutral", "positive"]
    print("=== VADER Classification Report ===")
    print(classification_report(y_true, vader_preds, target_names=class_names, zero_division=0))

    print("\n=== BERT Classification Report ===")
    print(classification_report(y_true, bert_preds, target_names=class_names, zero_division=0))

    print("\n=== Error Analysis: Neutral sentences ===")
    print("VADER correct, BERT wrong (neutral):")
    count = 0
    for i, (sent, true_label, v_pred, b_pred) in enumerate(zip(sentences, y_true, vader_preds, bert_preds)):
        if true_label == 1 and v_pred == 1 and b_pred != 1:
            print(f"Sentence {i}: {sent}")
            print(f"  True    : neutral")
            print(f"  VADER   : neutral")
            print(f"  BERT    : {class_names[b_pred]}")
            print("-" * 60)
            count += 1
            if count >= 5:
                break

    print("\nBERT correct, VADER wrong (neutral):")
    count = 0
    for i, (sent, true_label, v_pred, b_pred) in enumerate(zip(sentences, y_true, vader_preds, bert_preds)):
        if true_label == 1 and b_pred == 1 and v_pred != 1:
            print(f"Sentence {i}: {sent}")
            print(f"  True    : neutral")
            print(f"  VADER   : {class_names[v_pred]}")
            print(f"  BERT    : neutral")
            print("-" * 60)
            count += 1
            if count >= 5:
                break

    print("\n=== Example Disagreements (any label) ===")
    count = 0
    for i, (sent, true_label, v_pred, b_pred) in enumerate(zip(sentences, y_true, vader_preds, bert_preds)):
        if (v_pred != true_label) or (b_pred != true_label) or (v_pred != b_pred):
            print(f"Sentence {i}: {sent}")
            print(f"  True    : {class_names[true_label]}")
            print(f"  VADER   : {class_names[v_pred]}")
            print(f"  BERT    : {class_names[b_pred]}")
            print("-" * 60)
            count += 1
            if count >= 10:
                break

if __name__ == "__main__":
    main()
