#Game Classification Project
import pandas as pd
import re
import sklearn
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

# Load the dataset
df = pd.read_csv("steam_games.csv")

df["full_text"] = (
    df["title"].fillna("") + " " +
    df["tags"].fillna("") + " " +
    df["description"].fillna("")
)

df["primary_genre"] = df["genres"].apply(lambda x: str(x).split(",")[0].strip())

#train/test split
train_df,test_df = sklearn.model_selection.train_test_split(df, 
                                                            test_size=0.2, 
                                                            random_state=42,
                                                            stratify=df["primary_genre"])

#tokenizer 
tokenizer= BertTokenizer.from_pretrained("bert-base-uncased")
def process_data(texts):
    return tokenizer(list(texts), 
                     padding="max_length",
                     max_length=260, 
                     truncation=True,
                    return_tensors="pt")

#Dataset Class
class GameDataSet(Dataset):
    def __init__(self,df,tokenizer):
        self.df=df
        self.tokenizer=tokenizer
        self.texts=df["full_text"].tolist()
        self.labels=df["primary_genre"].tolist()

    def __getitem__(self,idx):
        text=self.texts[idx]
        label=self.labels[idx]
        encoding=self.tokenizer(text, 
                                padding="max_length", 
                                max_length=260, 
                                truncation=True, 
                                return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
}
    def __len__(self):
        return len(self.df)
#model
num_labels=df["primary_genre"].nunique()
model=BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

genre_mapping = dict(
    enumerate(df["primary_genre"].astype("category").cat.categories)
)
#prediction function
def predict_genre(user_input):
    # preprocess the input
    input_text = user_input.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)
    inputs =tokenizer(
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

# Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

   
#user interaction
if __name__ == "__main__":
    print("\n Model ready. Type a keyword or description.")
    while True:
        user_input = input("\nSearch (or 'quit'): ")
        if user_input.lower() == "quit":
            break

        genre = predict_genre(user_input)
        print(f"\nPredicted genres: {genre}")

        # Show some matching games
        matches = df[df["genres"].str.contains(genre, case=False, na=False)]
        print("\nGames that match your search:\n")
        for _, row in matches.head(5).iterrows():
            print(f"Title: {row['title']}")
            print(f"Tags: {row['tags']}")
            print(f"Description: {row['description'][:200]}...")
            print("-" * 40)
