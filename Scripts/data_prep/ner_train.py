from datasets import load_dataset
from simpletransformers.ner import NERModel, NERArgs
from pathlib import Path
from sklearn.metrics import classification_report
import pandas as pd
import torch
print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def main_train_ner():
    ontonote_types = [
        "CARDINAL", "DATE", "PERSON", "NORP", "GPE", "LAW", "PERCENT", "ORDINAL",
        "MONEY", "WORK_OF_ART", "FAC", "TIME", "QUANTITY", "PRODUCT", "LANGUAGE",
        "ORG", "LOC", "EVENT"
    ]
    ontonote_labels = ["O"]
    for t in ontonote_types:
        ontonote_labels += [f"B-{t}", f"I-{t}"]

    onto = load_dataset("tner/ontonotes5")["train"]
    rows = []
    for sent_id, ex in enumerate(onto):
        tokens = ex["tokens"]
        tag_ids = ex["tags"]
        for tok, tid in zip(tokens, tag_ids):
            rows.append({
                "sentence_id": sent_id,
                "words":       tok,
                "labels":      ontonote_labels[tid],
            })
    train_df = pd.DataFrame(rows)
    onto_val = load_dataset("tner/ontonotes5")["validation"]
    val_rows = []
    for sent_id, ex in enumerate(onto_val):
        for tok, tid in zip(ex["tokens"], ex["tags"]):
            val_rows.append({
                "sentence_id": sent_id,
                "words":       tok,
                "labels":      ontonote_labels[tid],
            })
    val_df = pd.DataFrame(val_rows)

    test_df = pd.read_csv("../../Files/NER-test.tsv", sep="\t", encoding="utf-8")
    test_df = test_df.rename(columns={"token": "words", "BIO_NER_tag": "labels"})
    test_df = test_df[["sentence_id", "words", "labels"]]
    test_df["labels"] = test_df["labels"].str.replace("LOCATION", "LOC")

    args = NERArgs()
    args.use_crf = True
    args.max_seq_length = 128
    args.train_batch_size = 16
    args.num_train_epochs = 6
    args.labels_list = ontonote_labels
    args.evaluate_during_training = True
    args.output_dir = "../../Files/Outputs"

    model = NERModel(
        "bert",
        "bert-base-cased",
        args=args,
        use_cuda=True
    )

    model.train_model(train_df, eval_data=val_df)
    result, model_outputs, wrong_preds = model.eval_model(test_df)
    print(result)


def main_eval_only():
    # 1) Rebuild your test_df exactly as before:
    test_df = pd.read_csv("../../Files/NER-test.tsv", sep="\t", encoding="utf-8")
    test_df = test_df.rename(columns={"token": "words", "BIO_NER_tag": "labels"})
    test_df = test_df[["sentence_id", "words", "labels"]]
    # Map "LOCATION" → "LOC" so label names match OntoNotes:
    test_df["labels"] = test_df["labels"].str.replace("LOCATION", "LOC")

    # 2) Re-create the same label list you used during training:
    ontonote_types = [
        "CARDINAL", "DATE", "PERSON", "NORP", "GPE", "LAW", "PERCENT", "ORDINAL",
        "MONEY", "WORK_OF_ART", "FAC", "TIME", "QUANTITY", "PRODUCT", "LANGUAGE",
        "ORG", "LOC", "EVENT"
    ]
    ontonote_labels = ["O"]
    for t in ontonote_types:
        ontonote_labels += [f"B-{t}", f"I-{t}"]

    # 3) Rebuild the same args (they have to match what you trained with)
    args = NERArgs()
    args.max_seq_length           = 128
    args.train_batch_size         = 16
    args.num_train_epochs         = 3
    args.labels_list              = ontonote_labels
    args.evaluate_during_training = True
    args.output_dir               = "../../Files/Outputs"

    # 4) Load the already-trained model from output_dir
    model = NERModel(
        "bert",
        "../../Files/Outputs",   # path where model.safetensors & config.json live
        args=args,
        use_cuda=False           # or True, if your GPU is available
    )

    # 5) Run evaluation (returns 3 items: result dict, flat outputs, wrong_predictions)
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    print("Overall NER metrics (from result):", result)
    print()

    # 6) Reconstruct token-level sentences and gold/predicted label sequences
    sentences = []   # list of token lists, one sub-list per sentence
    gold_seqs = []   # list of gold-label lists, one sub-list per sentence
    pred_seqs = []   # list of predicted-label lists, one sub-list per sentence

    for sent_id, group in test_df.groupby("sentence_id"):
        toks     = list(group["words"])
        gold_seq = list(group["labels"])
        pred_seq = model_outputs[sent_id]  # aligned by sentence_id

        sentences.append(toks)
        gold_seqs.append(gold_seq)
        pred_seqs.append(pred_seq)

    # 7) Print first 3 mis-predicted sentences (tokens / gold / pred)
    print("First 3 mis-predicted sentences:")
    for i in range(min(3, len(wrong_predictions))):
        print(f"Sentence {i}:")
        print("  Tokens:      ", sentences[i])
        print("  Gold Labels: ", gold_seqs[i])
        print("  Predicted:   ", wrong_predictions[i])
        print("-" * 60)
    print()

    # 8) Flatten all token-level labels to compute a full classification report
    flat_gold = []
    flat_pred = []
    # Here we explicitly iterate over each sequence and then each label
    for gold_seq, pred_seq in zip(gold_seqs, pred_seqs):
        for gl in gold_seq:
            flat_gold.append(gl)
        for pl in pred_seq:
            flat_pred.append(pl)

    # 9) Map each label string to an integer index (0,1,2,...)
    label2idx = {lab: i for i, lab in enumerate(ontonote_labels)}
    # Convert the flattened lists of strings to lists of ints
    y_true = [label2idx[lab] for lab in flat_gold]
    y_pred = [label2idx[lab] for lab in flat_pred]

    # 10) Print a classification_report over all labels 0..len(ontonote_labels)-1
    #     zero_division=0 ensures "0/0" cases show as 0.00 instead of throwing an error
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ontonote_labels))),
        zero_division=0
    )
    print("Token-level classification report (labels as integers):")
    print(report)



if __name__ == "__main__":
    main_train_ner()


