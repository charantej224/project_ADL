from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

lemma = WordNetLemmatizer()

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
department_classification = os.path.join(root_dir, "dept_classification/dept_classification.csv")
problem_classification = os.path.join(root_dir, "problem_classification/problem_classification.csv")

stopwords = stopwords.words("english")


def processing_text_lemmatize(input_text):
    cleaned_words = []
    tokens = word_tokenize(input_text)
    for token in tokens:
        if token not in stopwords:
            token = lemma.lemmatize(token, pos="v")
            cleaned_words.append(token)
    twitter_tweet = " ".join(cleaned_words)
    return twitter_tweet


# training a KNN classifier

classify_df = pd.read_csv(department_classification)
classify_df = classify_df.dropna()

classify_df.rename(columns={"DESCRIPTION": "desc", "DEPARTMENT_ID": "label"}, inplace=True)
classify_df["desc"].astype(str)
classify_df["desc"] = classify_df["desc"].apply(processing_text_lemmatize)

neighbors = len(list(classify_df.label.unique()))

tfidf_Vect = TfidfVectorizer()
X_features = tfidf_Vect.fit_transform(classify_df.desc)
Y_target = classify_df.label

x_train, x_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=neighbors).fit(x_train, y_train)

# accuracy on X_test
accuracy = knn.score(x_test, y_test)
print(accuracy)

# # creating a confusion matrix
# knn_predictions = knn.predict(x_test)
# cm = confusion_matrix(y_test, knn_predictions)
