import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

for token in doc:
    print(token.text)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

# Perform POS tagging
for token in doc:
    print(token.text, token.pos_)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)



# Lemmatization
for token in doc:
    print(token.text, token.lemma_)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)


# Phrase extraction
for chunk in doc.noun_chunks:
    print(chunk.text)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

# Text normalization
normalized_text = ' '.join([token.lower_ for token in doc if not token.is_punct])
print(normalized_text)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

# Stop word removal
filtered_text = ' '.join([token.text for token in doc if not token.is_stop])
print(filtered_text)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

# Dependency parsing
for token in doc:
    print(token.text, token.dep_)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)


# Word vector representation
for token in doc:
    print(token.text, token.vector)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

# Named entity recognition (NER)
for entity in doc.ents:
    print(entity.text, entity.label_)

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Tokenize the text
doc = nlp(text)

summary = ""

for sentence in doc.sents:
    summary += sentence.text + " "

print("Summary:", summary)

import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Open the text file
with open("test.txt", "r") as f:
    text = f.read()

# Process the text using the language model
doc = nlp(text)

# Count the number of words
num_words = len(doc)

# Print the result
print('Number of words:', num_words)

import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Open the text file
with open("test.txt", "r") as f:
    lines = f.readlines()

# Process each line using the language model
for line_num, line in enumerate(lines):
   
    
    # Count the number of words in the line
    num_words = len(line)
    
    # Print the result
    print(f'Line {line_num + 1}: Number of words = {num_words}')



# Load the requirements from the text file
with open('test.txt', 'r') as file:
    requirements = file.readlines()

# Load the requirement types and their corresponding labels
requirement_types = {
    'Functional (F)': ['functional', 'shall', 'should', 'must', 'able to'],
    'Availability (A)': ['availability', 'uptime', 'downtime', 'reliable', 'accessible'],
    'Fault Tolerance (FT)': ['fault tolerance', 'failover', 'redundancy', 'backup', 'recovery'],
    'Legal (L)': ['legal', 'regulation', 'compliance', 'govern', 'must comply with'],
    'Look & Feel (LF)': ['look and feel', 'user-friendly', 'intuitive', 'aesthetics'],
    'Maintainability (MN)': ['maintainability', 'modular', 'extensible', 'testable', 'reusable'],
    'Operational (O)': ['operational'],
    'Performance (PE)': ['performance'],
    'Portability (PO)': ['portability'],
    'Scalability (SC)': ['scalability'],
    'Security (SE)': ['security'],
    'Usability (US)': ['usability', 'user interface', 'user experience']
}

# Create a DataFrame with requirements and their corresponding labels
df = pd.DataFrame({'requirement': requirements})

# Create a new DataFrame to store the binary labels for each requirement type
label_df = pd.DataFrame()

# Assign labels to requirements based on keyword matching
for requirement_type, keywords in requirement_types.items():
    labels = df['requirement'].str.contains('|'.join(keywords), case=False)
    if labels.nunique() > 1:
        label_df[requirement_type] = labels

# Prepare the data for training
X = df['requirement']
y = label_df.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train separate classifiers for each requirement type
classifiers = {}
for i, requirement_type in enumerate(label_df.columns):
    classifier = SVC()
    classifier.fit(X_train_vectorized, y_train[:, i])
    classifiers[requirement_type] = classifier

# Predict the requirement types for the test set
y_pred = {}
for requirement_type, classifier in classifiers.items():
    y_pred[requirement_type] = classifier.predict(X_test_vectorized)

# Print the accuracy of each classifier
for requirement_type, y_pred_labels in y_pred.items():
    accuracy = (y_pred_labels == y_test[:, list(label_df.columns).index(requirement_type)]).mean()
    print(f'Accuracy for {requirement_type}: {accuracy}')