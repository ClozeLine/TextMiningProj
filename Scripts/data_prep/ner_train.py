from datasets import load_dataset
from simpletransformers.ner import NERModel, NERArgs
from pathlib import Path
import pandas as pd


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

    args = NERArgs()
    args.max_seq_length = 128
    args.train_batch_size = 16
    args.num_train_epochs = 3
    args.labels_list = ontonote_labels
    args.evaluate_during_training = True
    args.output_dir = "../../Files/Outputs"

    model = NERModel(
        "bert",
        "bert-base-cased",
        args=args,
        use_cuda=False
    )

    model.train_model(train_df, eval_data=val_df)


if __name__ == "__main__":
    main_train_ner()


