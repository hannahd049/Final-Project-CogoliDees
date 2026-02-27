#Game Classification Project
import pandas as pd
import re
import sklearn
from transformers import BertTokenizer

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
        

    def __getitem__(self,idx):

    def __len__(self):
        return len(self.df)
    

#user interaction search function
def predict_genre(user_input):
    # Preprocess the input
    input_text = user_input.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Predict the genre using the model
    

    # Map the predicted index to a genre label
    predicted_genre = genre_mapping[predicted_genre_idx]
    return predicted_genre      
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
