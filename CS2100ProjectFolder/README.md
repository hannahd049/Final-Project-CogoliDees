# Game Genre Classification

Created-by:
Hannah Dees and Sara Cogoli
Course: CS 2100

# Project-Overview:
This project uses machine learning to classify video games into genres such as Action, RPG, Puzzle, Horror, Adventure, Sports, Simulation, Shooter, and more. The model analyzes a game’s title, tags, and description to identify keywords strongly correlated with specific genres.
Users can enter any keyword or short phrase, and the system will:
- Predict the most likely genre based on the input
- Return a list of matching games from the dataset
- Display each game’s title, tags, and a short description
For example, searching for “zombies” will surface games commonly associated with the Horror genre.

# Goal
The overall goal for this project is to create a keyword-driven search tool that helps users discover games by concept rather than by name or popularity. This makes it easier to explore genres, find similar titles, and understand how text-based features can influence classification. 
# How it Works
- A BERT tokenizer converts text into model-ready input
- A machine learning classifier predicts the primary genre to display
- The system retrieves and displays games that match the predicted genre

# Key Techniques
- Text Processing
   - Combined multiple text fields into a single input string
   - Cleaned and normalized text
   - Tokenized text using BERT tokenizer
-Feature Extraction
   - Used BERT embeddings to generate contextual representations of each game
   - Explored Word2Vec as an alternate feature extraction method.
   - Compared different embedding strategies to see the effect on classification accuracy
-Classification
   - Trained and supervised a machine learning model to predict the primary game genre
   - Evaluated multiple classifiers
   - Used BERT-based embeddings to improve prediction quality
   - Measured performance accuracy, precision, and recall
-Clustering
   - Compared cluster grouping to actual genres to analyze model behavior
   - Used clustering to support keyword-based game recommendations
-Keyword Search
   - Implemented a simple keyword search tool
   - Mapped user keywords to the closest game genre using the classifier
   - Returned a list of games whose mebeddings are most similar to the user's input

# Dataset
The model is trained on a dataset of Steam games, including:
- Title
- Tags
- Descriptions
- Genres 
