import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


####### Function to read dataset from CSV files
def read_datasets():
    """ Reads users profile from CSV files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    
    # Create labels: 0 for fake users, 1 for genuine users
    x = pd.concat([genuine_users, fake_users])
    y = [1] * len(genuine_users) + [0] * len(fake_users)
    
    return x, y


####### Function for feature engineering
def extract_features(x):
    """ Extract relevant features from dataset """
    lang_list = list(enumerate(np.unique(x['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    
    # Map language to language code
    x['lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
    
    # Select important feature columns
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang_code']
    x = x.loc[:, feature_columns_to_use]
    
    return x


####### Function to plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """ Plot the learning curve for a given estimator """
    plt.figure()
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                     train_scores_mean + np.std(train_scores, axis=1), alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                     test_scores_mean + np.std(test_scores, axis=1), alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt


####### Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """ Plot the confusion matrix """
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


####### Function to plot ROC curve
def plot_roc_curve(y_test, y_pred):
    """ Plot ROC curve """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    
    print("False Positive Rate: ", false_positive_rate)
    print("True Positive Rate: ", true_positive_rate)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


####### Function to train and predict using Random Forest
def train(X_train, y_train, X_test):
    """ Train and predict using Random Forest classifier """
    clf = RandomForestClassifier(n_estimators=40, oob_score=True)
    clf.fit(X_train, y_train)
    
    print("The best classifier is: ", clf)
    
    # Estimate score using cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation scores: ", scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    
    # Plot learning curve
    title = 'Learning Curves (Random Forest)'
    plot_learning_curve(clf, title, X_train, y_train, cv=5)
    plt.show()
    
    # Predict on the test dataset
    y_pred = clf.predict(X_test)
    return y_pred


# Main process
if __name__ == "__main__":
    print("Reading datasets...\n")
    x, y = read_datasets()
    
    print("Extracting features...\n")
    x = extract_features(x)
    
    print("Splitting dataset into training and testing...\n")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
    
    print("Training datasets...\n")
    y_pred = train(X_train, y_train, X_test)
    
    print("Classification Accuracy on Test dataset: ", accuracy_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    
    # Classification report
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))
    
    # ROC curve
    plot_roc_curve(y_test, y_pred)