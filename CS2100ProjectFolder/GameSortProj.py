# Game Classification Project
import pandas as pd
import re
import sklearn
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

# Load the dataset
df = pd.read_csv("steam_games.csv")


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
        self.labels = df["primary_genre"].astype("category").cat.codes.tolist()

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

    return genre_mapping[predicted_genre_idx]

# User interaction
if __name__ == "__main__":
    print("\nModel ready. Type a keyword or description.")

    while True:
        user_input = input("\nSearch (or 'quit'): ")

        if user_input.lower() == "quit":
            break

        genre = predict_genre(user_input)
        print(f"\nPredicted genre: {genre}")

        # Show matching games
        matches = df[df["genre"].str.contains(genre, case=False, na=False)]

        print("\nGames that match your search:\n")
        for _, row in matches.head(5).iterrows():
            print(f"Title: {row['name']}")

            if "popular_tags" in row:
                print(f"Tags: {row['popular_tags']}")

            if "desc_snippet" in row:
                print(f"Description: {row['desc_snippet'][:200]}...")
            elif "game_description" in row:
                print(f"Description: {row['game_description'][:200]}...")
            else:
                print("Description: (none available)")

            print("-" * 40)


#blocked genres no nos
predicted_genre = model.predict(user_input)

if predicted_genre == "Sexual Content":
    predicted_genre = "Unknown"

print("Predicted genre:", predicted_genre)
