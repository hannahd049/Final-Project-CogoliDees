# Game Classification Project

#imports needed for our project
import pandas as pd
import re
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


# Load the dataset
df = pd.read_csv("steam_games.csv")

# Reduce dataset size to something trainable
df = df.sample(n=9000, random_state=42)    #CHANGE to test and TRAIN


# build text using columns that exist in OUR dataset
text_cols = [
    "name",
    "popular_tags",
    "desc_snippet",
    "game_description"
]

df["full_text"] = ""
for col in text_cols:
    if col in df.columns:
        df["full_text"] += df[col].fillna("") + " "

# Use the correct genre column
df["primary_genre"] = df["genre"].fillna("Unknown").apply(
    lambda x: str(x).split(",")[0].strip()
)

# Remove genres that appear only once
genre_counts = df["primary_genre"].value_counts()
df = df[df["primary_genre"].isin(genre_counts[genre_counts > 1].index)]

# Create shared category object for genres
genre_cat = df["primary_genre"].astype("category")
df["primary_genre"] = genre_cat
genre_mapping = dict(enumerate(genre_cat.cat.categories))

# Train/test split
train_df, test_df = sklearn.model_selection.train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def process_data(texts):
    return tokenizer(
        list(texts),
        padding="max_length",
        max_length=260,
        truncation=True,
        return_tensors="pt"
    )

# Dataset Class
class GameDataSet(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df["full_text"].tolist()
        self.labels = self.df["primary_genre"].cat.codes.tolist()

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=260,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

# Model
num_labels = df["primary_genre"].nunique()
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

# Genre mapping
genre_mapping = dict(
    enumerate(df["primary_genre"].astype("category").cat.categories)
)

# Prediction function
def predict_genre(user_input):
    input_text = user_input.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=260
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_genre_idx = torch.argmax(logits, dim=1).item()

    genre = genre_mapping[predicted_genre_idx]

    # Block unwanted genres
    blocked = {"Sexual Content", "Nudity", "Adult Only"}
    if genre in blocked:
        return "Unknown"

    return genre

#TRAINING CODE
train_dataset = GameDataSet(train_df, tokenizer)
train_loader = torch.utils.data.DataLoader(test_df, batch_size=16, shuffle=True)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader=torch.utils.data.DataLoader(GameDataSet(test_df, tokenizer), batch_size=16)

#move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Optimizer and scheduler
epochs=3
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

#TRAING MODEL CODE
def train_model():
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_postfix(loss=loss.item())
        print("\n Training Complete :)")

#validation accuracy code
def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            predicted_genre_idx = torch.argmax(logits, dim=1)
            correct += (predicted_genre_idx == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"\nValidation Accuracy: {accuracy:.4f}")

#start training and evaluating
print("Starting training...")
train_model()
evaluate_model()
print("Model ready!")

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Model saved!")

# User interaction
if __name__ == "__main__":
    print("\nModel ready. Type a keyword or description.")

    while True:
        user_input = input("\nSearch for a game using keywords or description (or 'quit'): ")

        if user_input.lower() == "quit":
            break

        genre = predict_genre(user_input)
        print(f"\nPredicted genre: {genre}")

        # Show matching games
        matches = df[df["genre"].str.contains(genre, case=False, na=False)]

        print("\nGames that match your search:\n")
        for _, row in matches.head(5).iterrows():
            print(f"Title: {row['name']}")

            if pd.notna(row.get("popular_tags", None)):
                print(f"Tags: {row['popular_tags']}")

            desc = None
            if pd.notna(row.get("desc_snippet", None)):
                desc = row["desc_snippet"]
            elif pd.notna(row.get("game_description", None)):
                desc = row["game_description"]

            if desc:
                print(f"Description: {desc[:200]}...")
            else:
                print("Description: (none available)")

            print("-" * 40)