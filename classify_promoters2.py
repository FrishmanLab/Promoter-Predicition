import sys
import numpy as np
import pandas as pd
import time
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classify_w_countvectorizer(true_promotors_file, decoy_promotors_file):
    true_sequences = []
    with open(true_promotors_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            true_sequences.append(line.strip())

    decoy_sequences = []
    with open(decoy_promotors_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            decoy_sequences.append(line.strip())

    allowed_letters = set('actgn')
    filtered_true_sequences = [seq for seq in true_sequences if set(seq.lower()) <= allowed_letters]
    filtered_decoy_sequences = [seq for seq in decoy_sequences if set(seq.lower()) <= allowed_letters]

    sequences = filtered_true_sequences + filtered_decoy_sequences
    labels = np.concatenate([np.ones(len(filtered_true_sequences)), np.zeros(len(filtered_decoy_sequences))])

    print(f'true_sequences = {len(true_sequences)}')
    print(f'filtered_true_sequences = {len(filtered_true_sequences)}')
    print(f'decoy_sequences = {len(decoy_sequences)}')
    print(f'filtered_decoy_sequences = {len(filtered_decoy_sequences)}')
    print("*********************************")

    k = #insert wanted k-length, 4 and 5 resulted in best results
    seqs = [' '.join([sequences[i][j:j+k] for j in range(0, len(sequences[i]), k)]) for i in range(len(sequences))]

    X_train, X_test, y_train, y_test = train_test_split(seqs, labels, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    print("X_train shape:", X_train.shape) 
    print("X_test shape:", X_test.shape) 
    print("*********************************")

    print("CountVectorizer + SGDClassifier")

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'loss': ['log_loss'],  
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000, 2000, 3000]
    }

    # Train the model with GridSearchCV
    sgd = SGDClassifier(random_state=42)
    grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_sgd = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    y_pred = best_sgd.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Best SGD model performance:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 score:", f1)

    joblib.dump((best_sgd, vectorizer), 'countvectorizer_sgd_best_clf_INSERTLENGTHOFKmer.pkl')
    print("Model and vectorizer saved as 'countvectorizer_sgd_best_clf_6mer.pkl'")
    print("*********************************")

def main(argv):
    true_promotors_file = argv[0]
    decoy_promotors_file = argv[1]
    classify_w_countvectorizer(true_promotors_file, decoy_promotors_file)

if __name__ == "__main__":
    main(sys.argv[1:])
