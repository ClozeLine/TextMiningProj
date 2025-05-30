from datasets import load_dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, TrainingArguments
import pandas as pd


def ner_training():
    # load train and val sets
    dataset = load_dataset("tner/ontonotes5")
    train_set = dataset["train"]
    val_set = dataset["validation"]

    label_list = dataset['train'].features['tags'].feature.names

    def transform_to_df(data):
        rows = []
        for item in data:
            for token, tag in zip(item['tokens'], item['tags']):
                rows.append({
                    'sentence_id': item['id'],
                    'words': token,
                    'labels': label_list[tag]
                })
        return

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased', num_labels=len(label_list))

    args = TrainingArguments(
        output_dir="./ner_bert",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    ner_training()
