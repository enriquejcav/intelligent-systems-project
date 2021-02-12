import os
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer

def dataframe_train_val():
    df = pd.read_csv(os.environ['DATASET_PATH'])
    return df


def dataframe_test():
    df_test = pd.read_csv(r'../data/test_products.csv')
    return df_test


def get_columns(dataframe, columns):
    restricted_columns_df = dataframe[columns]
    return restricted_columns_df


def calc_pairwise_similarity(lista):
    vectorizer = TfidfVectorizer(min_df=1)  # min_df ignora termos que aparecem em menos de 1 documento/ 1%, 19%
    tfidf = vectorizer.fit_transform(lista)
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity


def take_occurences(pairwise_similarity,i):
    similarity = pairwise_similarity[i].toarray()
    occurrences = np.count_nonzero(similarity >= 0.90)
    return occurrences, similarity


def max_similarity(pairwise_similarity):
    title_similarity = []
    for i in range(0, pairwise_similarity.shape[0]):
        occurences, similarity = take_occurences(pairwise_similarity, i)
        for j in range(0, similarity.shape[1]):
            position = similarity[0][j]
            if position >= 0.25:
                title_similarity.append(j)
                break
            elif occurences == 0:
                title_similarity.append(22214)
                break
        if i % 1000 == 0:
            print(f"We've already made {i} similaritys calculations.")

    print(f"We've already made finished in {i} similaritys calculations.")
    return title_similarity


def decoder_label_to_unique(label_encoder,label_decoder_with_duplicity): #X_train, X

    X_train_str = []

    for i in range(0, len(label_decoder_with_duplicity)):
        X_train_str.append(label_decoder_with_duplicity[label_encoder[i][0]][0])

    return X_train_str


def variabletest_to_labelled(X_train,X_train_str,X_test):
    X_test_labelled = []
    for j in range(0,len(X_test)):
        coef_similarity = []
        for i in range(0, X_train.shape[0]):
            callable_object_similarity = difflib.SequenceMatcher(None,X_train_str[i], X_test[j][0])
            similarity_test_with_each_train_classes = callable_object_similarity.ratio()
            coef_similarity.append(similarity_test_with_each_train_classes)
            #print(similarity_test_with_each_train_classes)

        X_test_labelled.append(X_train[coef_similarity.index(max(coef_similarity))][0])
        if j % 10 == 0:
            print(f"We've already made {j} similaritys calculations.")

    print(f"We've already made finished in {j} similaritys calculations.")
    return X_test_labelled