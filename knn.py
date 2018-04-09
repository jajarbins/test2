import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import pandas as pd


def preparing_range():
    k_range = list(range(1, 26))
    k_range_tampon = k_range.copy()
    for k in k_range_tampon:
        if k % 3 == 0:
            k_range.remove(k)
    print(k_range)
    return k_range


def loading_dataset():
    iris_scikit = datasets.load_iris()
    iris_pd = pd.DataFrame(iris_scikit.data, columns=iris_scikit.feature_names)
    iris_pd['target'] = pd.Series(iris_scikit.target)
    print(list(iris_pd.columns.values))
    return iris_pd


def adding_values(iris_pd):
    iris_pd.insert(loc=2, column='sepal ratio', value=iris_pd['sepal length (cm)'] / iris_pd['sepal width (cm)'])
    iris_pd.insert(loc=5, column='petal ratio', value=iris_pd['petal length (cm)'] / iris_pd['petal width (cm)'])
    iris_pd.insert(loc=6, column='sepal_petal ratio', value=iris_pd['sepal ratio'] / iris_pd['petal ratio'])
    print(list(iris_pd.columns.values))
    return iris_pd


def preparing_dataset():
    iris_pd = loading_dataset()
    return adding_values(iris_pd)


def preparing_dataframes_and_target(iris_pd):
    dataframe_tampon = iris_pd.drop(['target'], axis=1)
    list_dataframes = preparing_dataframes(preparing_combinaison(dataframe_tampon), dataframe_tampon)
    y = iris_pd['target']
    return list_dataframes, y


def preparing_combinaison(dataframe_tampon):
    combination = []
    for r in range(1, len(dataframe_tampon)):
        combination += list(itertools.combinations(dataframe_tampon.columns, r))
    #print(combination)
    return combination


def preparing_dataframes(combinaison, dataframe):
    list_dataframe = []
    for r in range(0, len(combinaison)):
        list_dataframe.append(dataframe.filter(items=combinaison[r]))
    #print(len(combinaison))
    #print(len(list_dataframe))
    return list_dataframe


def preparing_data(dataframe, y):
    X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2, random_state=5)
    return X_train, X_test, y_train, y_test


def knn_algo(X_train, X_test, y_train, y_test, k_range):
    scores = []
    for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))
    return scores


def plot_knn(k_range, scores):
    plt.plot(k_range, scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
    plt.show()


def index_of_max(list_scores):
    k_range = 27
    index = 0
    maximum = 0
    for r in range(0, len(list_scores)):
        max_tampon = max(list_scores[r])
        k_range_tampon = list_scores[r].index(max_tampon)
        if max_tampon == maximum:
            if k_range_tampon < k_range:
                maximum, index, k_range = storing_and_printing_index_of_max_values(max_tampon, r, k_range_tampon)
        if max_tampon > maximum:
                maximum, index, k_range = storing_and_printing_index_of_max_values(max_tampon, r, k_range_tampon)
    return index


def storing_and_printing_index_of_max_values(max_tampon, r, k_range_tampon):
    print("maximum : {}".format(max_tampon))
    print("index : {}".format(r))
    print("k_range : {}".format(k_range_tampon))
    return max_tampon, r, k_range_tampon


def printing_and_ploting_output_values(list_scores, list_dataframes, k_range):
    index_max_scored_list = index_of_max(list_scores)
    print(list_dataframes[index_max_scored_list].columns)
    plot_knn(k_range, list_scores[index_max_scored_list])


list_scores = []
k_range = preparing_range()
iris_pd = preparing_dataset()
list_dataframes, y = preparing_dataframes_and_target(iris_pd)
for i in range(0, len(list_dataframes)):
    X_train, X_test, y_train, y_test = preparing_data(list_dataframes[i], y)
    scores = knn_algo(X_train, X_test, y_train, y_test, k_range)
    list_scores.append(scores)
printing_and_ploting_output_values(list_scores, list_dataframes, k_range)