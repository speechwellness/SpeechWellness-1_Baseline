import os
import json
import argparse
import gc

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import (
    SchedulerType,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from tqdm import tqdm

from metric import Metrics
from text_model import TransformerWithHead
from text_dataset import TextDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google-bert/bert-base-chinese")
    parser.add_argument("--data_path", type=str, default="./transcriptions")
    parser.add_argument("--task", type=str, default="0")
    parser.add_argument(
        "--train_json",
        type=str,
        default="data/train_data.json",
    )
    parser.add_argument(
        "--dev_json",
        type=str,
        default="data/dev_data.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/task0/text1",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_label", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay to use.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=20,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    args = parser.parse_args()
    return args


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    os.system("cp {} {}".format("script/text/text_train.py", os.path.join(args.output_dir, "train.py")))
    os.system("cp {} {}".format("script/text/text_model.py", os.path.join(args.output_dir, "model.py")))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_dataset = TextDataset(args.train_json, tokenizer=tokenizer, args=args)
    dev_dataset = TextDataset(args.dev_json, tokenizer=tokenizer, args=args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = TransformerWithHead(
        args.model_path,
        linear_probe=False,
        num_label=args.num_label,
        dropout=args.dropout,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Metrics(num_classes=args.num_label)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    best_metric = None

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        predictions = []
        references = []
        for step, batch in enumerate(train_dataloader):
            labels = batch["labels"].to(device)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_masks"].to(device),
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            global_step += 1
            progress_bar.update(1)

            total_loss += loss.detach()
            running_loss += loss.detach()
            predictions.extend(torch.argmax(logits, dim=-1).cpu())
            references.extend(labels.cpu())
            writer.add_scalar('train/running_loss', running_loss, global_step=global_step)
            running_loss = 0

            if global_step % args.evaluation_steps == 0:
                eval_metrics = evaluation_loop(
                    model, dev_dataloader, metric, loss_fn, device
                )
                print(f"Valid metrics in step {global_step}: {eval_metrics}")
                writer.add_scalar('valid/loss', eval_metrics["loss"], global_step=global_step)
                writer.add_scalar('valid/accuracy', eval_metrics["accuracy"], global_step=global_step)
                writer.add_scalar('valid/f1', eval_metrics["f1"], global_step=global_step)
                writer.add_scalar('valid/precision', eval_metrics["precision"], global_step=global_step)
                writer.add_scalar('valid/recall', eval_metrics["recall"], global_step=global_step)

                if best_metric is None or eval_metrics["accuracy"] >= best_metric:
                    best_metric = eval_metrics["accuracy"]
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        train_metrics = metric.compute(predictions, references)
        writer.add_scalar('train/accuracy', train_metrics["accuracy"], global_step=epoch)
        writer.add_scalar('train/f1', train_metrics["f1"], global_step=epoch)
        writer.add_scalar('train/precision', train_metrics["precision"], global_step=epoch)
        writer.add_scalar('train/recall', train_metrics["recall"], global_step=epoch)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))


def evaluation_loop(model, dataloader, metric, loss_fn, device):
    model.eval()
    predictions = []
    references = []
    total_loss = 0
    eval_step = 0
    for n, batch in enumerate(dataloader):
        with torch.no_grad():
            labels = batch["labels"].to(device)
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_masks"].to(device),
            )
            total_loss += loss_fn(preds, labels)
            preds = preds.cpu()
            labels = labels.cpu()
            predictions.extend(torch.argmax(preds, dim=-1))
            references.extend(labels)
            eval_step += 1
        del batch
        gc.collect()
    model.train()
    eval_metrics = metric.compute(predictions=predictions, references=references)
    eval_metrics.update({"loss": total_loss / eval_step})
    return eval_metrics

if __name__ == "__main__":
    main()