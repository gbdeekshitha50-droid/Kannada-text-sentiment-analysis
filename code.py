#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install pandas numpy scikit-learn')
get_ipython().system('pip install matplotlib seaborn')
get_ipython().system('pip install nltk openpyxl')
get_ipython().system('pip install streamlit plotly joblib')
get_ipython().system('pip install wordcloud')


# In[6]:


import pandas as pd
import numpy as np
import sklearn

print("All libraries imported successfully!")


# In[7]:


import os
import pandas as pd
import numpy as np

# Dataset folder location
DATASET_PATH = "datasets"

# Create an empty dictionary
datasets = {}

print("Initialization completed!")
print("Current Directory:", os.getcwd())
print("Dataset Path:", DATASET_PATH)


# In[8]:


files = {
    "mobile": "mobile_preprocessed.xlsx",
    "earphones": "earphones_preprocessed.xlsx",
    "laptop": "laptop_preprocessed.xlsx",
    "mouse": "mouse_preprocessed.xlsx",
    "speaker": "speaker_preprocessed.xlsx"
}

for name, filename in files.items():

    path = os.path.join(DATASET_PATH, filename)

    if os.path.exists(path):

        df = pd.read_excel(path)
        datasets[name] = df

        print(f"\n{name} Dataset")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        display(df.head(3))

    else:
        print(f"\n{name}: FILE NOT FOUND")


# In[9]:


import pandas as pd
import numpy as np
import re
import unicodedata
import os

print("Preprocessing libraries imported successfully!")


# In[10]:


kannada_stopwords = {
    "ಮತ್ತು",
    "ಆದರೆ",
    "ಈ",
    "ಆ",
    "ಒಂದು",
    "ಇದು",
    "ಅದು",
    "ಅವರು",
    "ನಾವು",
    "ನಾನು",
    "ಇದೆ",
    "ಇತ್ತು"
}
def preprocess_kannada(text):

    if pd.isna(text):
        return ""

    # Step 1: Convert to string
    text = str(text)

    # Step 2: Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 4: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Step 5: Remove unwanted symbols
    text = re.sub(
        r'[^\u0C80-\u0CFFa-zA-Z0-9\s]',
        ' ',
        text
    )

    # Step 6: Convert English text to lowercase
    text = text.lower()

    # Step 7: Tokenization
    tokens = text.split()

    # Step 8: Stopword removal
    tokens = [
        word for word in tokens
        if word not in kannada_stopwords
    ]

    # Step 9: Join tokens
    processed_text = " ".join(tokens)

    return processed_text.strip()


print("Kannada preprocessing function created!")
for name, df in datasets.items():

    df["processed_text"] = df["clean_text"].apply(
        preprocess_kannada
    )

    print("\nDataset:", name)
    print("Original text:")
    print(df["clean_text"].iloc[0])

    print("\nProcessed text:")
    print(df["processed_text"].iloc[0])

    print("\nPreprocessing completed!")


# In[11]:


os.makedirs("datasets/processed", exist_ok=True)

for name, df in datasets.items():

    filename = name.lower() + "_processed.csv"

    df.to_csv(
        os.path.join("datasets/processed", filename),
        index=False,
        encoding="utf-8-sig"
    )

    print(name, "saved successfully!")
for name, df in datasets.items():

    print("\n", name)

    print("Total records:", len(df))

    print(
        "Empty processed texts:",
        (df["processed_text"].str.strip() == "").sum()
    )

    print(
        "Duplicate processed texts:",
        df["processed_text"].duplicated().sum()
    )

    print(
        "Sentiment distribution:"
    )

    print(df["sentiment"].value_counts())


# In[12]:


import os
import pandas as pd

print("Available Excel files:")

for file in os.listdir("datasets"):
    if file.endswith(".xlsx"):
        print(file)


# In[13]:


import pandas as pd

combined_data = []

for name, df in datasets.items():

    # Identify the available text column
    possible_text_columns = [
        "processed_text",
        "clean_text",
        "text",
        "Kannada_Translation",
        "sentence"
    ]

    text_col = next(
        (col for col in possible_text_columns if col in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(f"No text column found in {name}")

    if "sentiment" not in df.columns:
        raise ValueError(f"Sentiment column missing in {name}")

    temp = df[[text_col, "sentiment"]].copy()

    temp.columns = ["text", "sentiment"]
    temp["dataset"] = name

    combined_data.append(temp)

# Combine all five datasets
combined_df = pd.concat(combined_data, ignore_index=True)

# Remove missing and empty records
combined_df.dropna(subset=["text", "sentiment"], inplace=True)
combined_df["text"] = combined_df["text"].astype(str).str.strip()
combined_df = combined_df[combined_df["text"] != ""]

combined_df.reset_index(drop=True, inplace=True)

print("Combined dataset shape:", combined_df.shape)
print("\nDataset-wise record count:")
print(combined_df["dataset"].value_counts())

print("\nOverall sentiment distribution:")
print(combined_df["sentiment"].value_counts())

display(combined_df.head(10))

import matplotlib.pyplot as plt

combined_df["sentiment"].value_counts().plot(
    kind="bar",
    figsize=(7, 4)
)

plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()



# In[14]:


# Keep only the five selected datasets
selected_datasets = [
    "earphones",
    "laptop",
    "mobile",
    "mouse",
    "speaker"
]

combined_data = []

for name in selected_datasets:
    df = datasets[name].copy()

    text_col = next(
        col for col in [
            "processed_text", "clean_text", "text",
            "Kannada_Translation", "sentence"
        ]
        if col in df.columns
    )

    temp = df[[text_col, "sentiment"]].copy()
    temp.columns = ["text", "sentiment"]
    temp["dataset"] = name.capitalize()

    combined_data.append(temp)

combined_df = pd.concat(combined_data, ignore_index=True)

combined_df.dropna(subset=["text", "sentiment"], inplace=True)
combined_df["text"] = combined_df["text"].astype(str).str.strip()
combined_df = combined_df[combined_df["text"] != ""]
combined_df.reset_index(drop=True, inplace=True)

print("Corrected combined dataset:", combined_df.shape)
print("\nDataset distribution:")
print(combined_df["dataset"].value_counts())

print("\nSentiment distribution:")
print(combined_df["sentiment"].value_counts())

display(combined_df.head())


# In[15]:


from sklearn.model_selection import train_test_split

X = combined_df["text"]
y = combined_df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nTesting sentiment distribution:")
print(y_test.value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    token_pattern=r"(?u)\b\w+\b"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
print("Total vocabulary features:", len(tfidf.vocabulary_))



# In[16]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import pandas as pd

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        random_state=42,
        class_weight="balanced"
    )
}

trained_models = {}
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    trained_models[name] = model
    predictions[name] = y_pred

    print(f"{name} training completed!")


results = []

for name, y_pred in predictions.items():

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("MODEL PERFORMANCE COMPARISON")
display(results_df.round(4))

for name, y_pred in predictions.items():

    print("\n" + "=" * 50)
    print("MODEL:", name)
    print("=" * 50)

    print(classification_report(
        y_test,
        y_pred,
        labels=["positive", "neutral", "negative"],
        zero_division=0
    ))





# In[17]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

for name, y_pred in predictions.items():

    print("\nConfusion Matrix:", name)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=["positive", "neutral", "negative"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()


# In[ ]:




