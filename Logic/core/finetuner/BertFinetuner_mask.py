import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login, create_repo
import json
import os
from collections import Counter


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres)
        self.df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.X_test = None
        self.y_test = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as f:
            self.df = json.load(f)
        print("Dataset loaded successfully.")

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        genres = [entry['genres'] for entry in self.df]
        genre_counts = Counter([genre for sublist in genres for genre in sublist])
        top_genres = [genre for genre, _ in genre_counts.most_common(self.top_n_genres)]

        filtered_dataset = []
        for entry in self.df:
            entry_genres = [genre for genre in entry['genres'] if genre in top_genres]
            if entry_genres:
                entry['genres'] = entry_genres
                filtered_dataset.append(entry)

        self.df = filtered_dataset

        top_genre_counts = {genre: genre_counts[genre] for genre in top_genres}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(top_genre_counts.keys()), y=list(top_genre_counts.values()))
        plt.title('Top Genres Distribution')
        plt.xlabel('Genres')
        plt.ylabel('Frequency')
        plt.show()

    def split_dataset(self, test_size=0.2, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        df = pd.DataFrame(self.df)
        X = df["first_page_summary"].tolist()
        y = df['genres'].str[0].tolist()
        y = LabelEncoder().fit_transform(y)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

        self.train_data = self.create_dataset(X_train, y_train)
        self.val_data = self.create_dataset(X_val, y_val)
        self.test_data = self.create_dataset(X_test, y_test)
        self.X_test = X_test
        self.y_test = y_test

        print("Dataset split into train, validation, and test sets.")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        encodings = self.tokenizer(list(map(str, encodings)), truncation=True, padding=True)
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model = trainer

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        raw_pred, _, _ = self.model.predict(self.test_data)
        y_pred = np.argmax(raw_pred, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
        acc = accuracy_score(self.y_test, y_pred)
        evaluation_metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        print(evaluation_metrics)

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_model(model_name)
        self.tokenizer.save_pretrained(model_name)
        token = "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"
        login(token)
        repo_url = create_repo(repo_id=model_name)

        self.model.push_to_hub(model_name, token)
        self.tokenizer.push_to_hub(model_name, token)

        print(f"Model saved.")


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)