from pydotplus import graphviz
from sklearn import tree
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from IPython.display import Image, display
import matplotlib.pyplot as plt

def load_data_set():
    df = pd.read_csv('playtennis.csv') 
    lb = preprocessing.LabelEncoder()
    df = df.apply(lb.fit_transform)
    feature_cols = ['Outlook','Temprature','Humidity','Wind']
    X = df[feature_cols] 
    y = df.Play_Tennis 
    return df, X, y

def train_model(data, X, y):
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X, y)
    X_pred = clf.predict(X)
    return clf, X_pred

def display_image(decision_tree_classifier, iris_data, save=False):
    text_representation = tree.export_text(decision_tree_classifier)
    fig = plt.figure(figsize=(5,5))
    _ = tree.plot_tree(decision_tree_classifier,feature_names= ['Outlook','Temprature','Humidity','Wind'],filled=False,fontsize=10)
    if save:
        fig.savefig("/Users/salvatoreprioli/Documents/ML_/Machine-Learning/DecisionTrees/decistion_tree.png")
    else:
        plt.show()
 
if __name__ == '__main__':
    data, X, y = load_data_set()
    decision_tree_classifier, X_pred = train_model(data, X, y)
    display_image(decision_tree_classifier, data)
    print(X_pred==y)

