# coding: utf-8

### Detect fake profiles in online social networks using Support Vector Machine

import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC

####### function for reading dataset from csv files

def read_datasets():
    """ Reads users profile from csv files """
    try:
        genuine_users = pd.read_csv("data/users.csv")
        print("Genuine users dataset loaded successfully.")
        fake_users = pd.read_csv("data/fusers.csv")
        print("Fake users dataset loaded successfully.")
        
        x = pd.concat([genuine_users, fake_users])
        print(f"Concatenated dataset shape: {x.shape}")
        
        y = len(fake_users) * [0] + len(genuine_users) * [1]
        return x, y
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

####### function for sampling the dataset

def sample_dataset(x, y, frac=0.1, random_state=42):
    """ Randomly sample a fraction of the dataset """
    try:
        x_sampled = x.sample(frac=frac, random_state=random_state)
        y_sampled = np.array(y)[x_sampled.index]
        print(f"Sampled dataset shape: {x_sampled.shape}")
        return x_sampled, y_sampled
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        return x, y

####### function for predicting sex using name of person

def predict_sex(name):
    try:
        d = gender.Detector()
        first_name = name.split()[0]
        sex = d.get_gender(first_name)
        sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
        sex_code = sex_dict.get(sex, 0)
        print(f"Name: {name}, Predicted Sex: {sex_code}")
        return sex_code
    except Exception as e:
        print(f"Error predicting gender for name {name}: {e}")
        return 0  # Return a default value in case of error

####### function for feature engineering

def extract_features(x):
    print("Starting feature extraction...")
    try:
        # Language encoding
        lang_list = list(enumerate(np.unique(x['lang'])))
        lang_dict = {name: i for i, name in lang_list}
        x['lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
        print("Language feature encoded.")
        
        # Predict sex and add to the dataset
        x['sex_code'] = x['name'].apply(predict_sex)
        print("Sex feature encoded.")
        
        # Select relevant features
        feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 
                                  'favourites_count', 'listed_count', 'sex_code', 'lang_code']
        x = x.loc[:, feature_columns_to_use]
        print("Features selected.")
        
        return x
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

####### function for plotting learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

####### function for plotting confusion matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['Fake', 'Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####### function for plotting ROC curve

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    print("False Positive rate: ", false_positive_rate)
    print("True Positive rate: ", true_positive_rate)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

####### Function for training data using Support Vector Machine

def train(X_train, y_train, X_test):
    """ Trains and predicts dataset with a SVM classifier """
    # Scaling features
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    Cs = 10.0 ** np.arange(-2, 3, .5)
    gammas = 10.0 ** np.arange(-2, 3, .5)
    param = [{'gamma': gammas, 'C': Cs}]
    cvk = StratifiedKFold(n_splits=5)
    classifier = SVC()
    clf = GridSearchCV(classifier, param_grid=param, cv=cvk)
    clf.fit(X_train, y_train)
    print("The best classifier is: ", clf.best_estimator_)
    clf.best_estimator_.fit(X_train, y_train)
    # Estimate score
    scores = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5)
    print(scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    title = 'Learning Curves (SVM, rbf kernel, $\gamma=%.6f$)' % clf.best_estimator_.gamma
    plot_learning_curve(clf.best_estimator_, title, X_train, y_train, cv=5)
    plt.show()
    # Predict class
    y_pred = clf.best_estimator_.predict(X_test)
    return y_test, y_pred

# Main execution starts here

print("reading datasets.....\n")
x, y = read_datasets()

# Sample the dataset (adjust the frac parameter to sample more or less data)
x, y = sample_dataset(x, y, frac=0.1)

print("extracting features.....\n")
x = extract_features(x)
print(x.columns)
print(x.describe())

print("splitting datasets into train and test dataset...\n")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

print("training datasets.......\n")
y_test, y_pred = train(X_train, y_train, X_test)

print('Classification Accuracy on Test dataset: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

plot_roc_curve(y_test, y_pred)
