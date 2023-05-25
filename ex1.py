import argparse
import numpy as np
import os
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
    default_data_collator,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
#import wandb
import subprocess
import torch

os.environ['TRANSFORMERS_CACHE'] ='.transformers_cache'
os.environ['HF_HOME'] = '.hf_home'
os.environ['HF_DATASETS_CACHE'] = '.hf_datasets_cache'

import random


# Helper function to compute accuracy
def compute_accuracy(preds, labels):
    return np.mean(preds == labels)

# Function to generate requirements.txt file
def generate_requirements():
    # Run pipreqs to generate requirements.txt
    subprocess.run(["pipreqs", "."])

    # Move the generated requirements.txt to the correct location
    #os.rename("requirements.txt", "code/requirements.txt")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": compute_accuracy(predictions, labels)}

# Function to fine-tune a specific model
def fine_tune_model(model_name, num_seeds, num_train_samples, num_val_samples):
    # Initialize lists to store accuracies
    accuracies = []


    for seed in range(num_seeds):
        # start a new wandb run to track this script
        #wandb.init(project="anlp_ex1", config={"dataset": "SST2", }, name=f"{model_name}_seed_{seed}")

        # Set the seed for reproducibility
        set_seed(seed)

        # Load the dataset
        dataset = load_dataset("sst2")
        if num_train_samples == -1:
            num_train_samples = len(dataset["train"])
            print("num_train_samples", num_train_samples)

        train_dataset = dataset["train"][:num_train_samples]
        if num_val_samples == -1:
            num_val_samples = len(dataset["validation"])
            print("num_val_samples", num_val_samples)
        val_dataset = dataset["validation"][:num_val_samples]

        # Tokenize the input sequences
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(train_dataset["sentence"])
        val_encodings = tokenizer(val_dataset["sentence"])

        # Prepare the datasets
        train_dataset = SentimentDataset(train_encodings, train_dataset["label"])
        val_dataset = SentimentDataset(val_encodings, val_dataset["label"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Define the model and training arguments
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        training_args = TrainingArguments(
            output_dir=f"{model_name}_seed_{seed}",
            evaluation_strategy="epoch",
            seed=seed,
            report_to=["none"],
            save_strategy="no",
            logging_steps=10,

        )

        # Define the trainer and train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        # Start the training
        trainer.train()

        name_witout_slash = model_name.replace("/", "_")
        trainer.save_model(f"{name_witout_slash}_seed_{seed}")

        # Evaluate the model on the validation set
        eval_result = trainer.evaluate(eval_dataset=val_dataset)
        accuracy = eval_result["eval_accuracy"]
        accuracies.append(accuracy)

        #wandb.finish()

    return accuracies



# Dataset class for sentiment analysis
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune language models for sentiment analysis on SST2 dataset.")
    parser.add_argument("num_seeds", type=int, help="Number of seeds to be used for each model.", default=3, nargs="?")
    parser.add_argument("num_train_samples", type=int, help="Number of samples to be used during training.", default=-1, nargs="?")
    parser.add_argument("num_val_samples", type=int, help="Number of samples to be used during validation.", default=-1, nargs="?")
    parser.add_argument("num_test_samples", type=int,
                        help="Number of samples for which the model will predict a sentiment.", default=-1, nargs="?")
    return parser.parse_args()

def test(best_model_name, best_seed, num_test_samples):
    # Select the model with the highest mean accuracy
    print(f"Best model: {best_model_name} with seed {best_seed}")

    # Load the best model
    name_witout_slash = best_model_name.replace("/", "_")
    best_model = AutoModelForSequenceClassification.from_pretrained(f"{name_witout_slash}_seed_{best_seed}")

    # Load the test dataset
    dataset = load_dataset("sst2")
    if num_test_samples == -1:
        num_test_samples = len(dataset["test"])
    test_dataset = dataset["test"][:num_test_samples]

    # Tokenize the input sequences for testing
    tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    test_encodings = tokenizer(test_dataset["sentence"])

    # Prepare the test dataset
    encoded_test_dataset = SentimentDataset(test_encodings, test_dataset["label"])

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the trainer and compute predictions
    trainer = Trainer(model=best_model,
                      data_collator=data_collator,
                      )

    start_time = time.time()
    predictions = trainer.predict(encoded_test_dataset)
    predict_time = time.time() - start_time

    # Save the predictions to a file
    with open("predictions.txt", "w") as f:
        for idx, prediction in enumerate(predictions.predictions.argmax(axis=1)):
            sentence = test_dataset["sentence"][idx]
            f.write(f"{sentence}###{prediction}\n")

    with open("res.txt", "a") as f:
        # write predict time to file
        f.write(f"predict time,{predict_time}\n")


# Main function
def main(num_seeds, num_train_samples, num_val_samples, num_test_samples):

    # Define the models and their corresponding tokenizer classes
    models = ["bert-base-uncased","roberta-base","google/electra-base-generator"]

    # Initialize lists to store results
    accuracies = []

    start_time = time.time()
    for model_name in models:

        # Fine-tune the model and compute accuracies
        print(f"Fine-tuning {model_name}")
        model_accuracies = fine_tune_model(model_name, num_seeds, num_train_samples, num_val_samples)

        # Compute mean and standard deviation of accuracies
        mean_accuracy = np.mean(model_accuracies)
        std_accuracy = np.std(model_accuracies)

        # Print and log the results
        print(f"{model_name}: Mean Accuracy: {mean_accuracy:.4f} +- {std_accuracy:.4f}")
        accuracies.append((model_name, mean_accuracy, std_accuracy))

    train_time = time.time() - start_time

    # Write the results to the res.txt file
    with open("res.txt", "w") as f:
        for accuracy in accuracies:
            f.write(f"{accuracy[0]},{accuracy[1]:.4f} +- {accuracy[2]:.4f}\n")

    with open("res.txt", "a") as f:
        # Write the train time to the res.txt file
        f.write(f"train time,{train_time}\n")


    # Sort the accuracies in descending order based on mean accuracy
    accuracies.sort(key=lambda x: x[1], reverse=True)
    best_model_name = accuracies[0][0]
    best_seed = np.argmax([acc[1] for acc in accuracies])
    test(best_model_name, best_seed, num_test_samples)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Call the function to generate requirements.txt
    #generate_requirements()

    # Run the main function
    main(args.num_seeds, args.num_train_samples, args.num_val_samples, args.num_test_samples)
