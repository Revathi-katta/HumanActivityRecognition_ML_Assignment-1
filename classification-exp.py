import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
#Splitting the data into training(70%) and testing(30%)
train_size=int(0.7*len(x))
X_trian_data,X_test_data=X[:train_size],X[trian_size:]
Y_train_data,Y_test_data=Y[:train_size],Y[train_size:]

#trianing the decisiontree model
tree=DecisiontreeClassifier()
tree.fit(X_train,Y_train)

#making predictions on the test set
Y_pred=tree.predict(X_test)

#metrics calculation
accuracy=accuracy_score(Y_test,Y_pred)
precision=precision_score(Y_test,Y_pred)
recall=recall_score(Y_test,Y_pred)
print(f"Accuracy:{accuracy:.2f}")
print(f"Precision:{precision}")
print(f"Recall:{recall}")

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

k=5
predictoins = {}
accuracies=[]
Kf = KFold(n_splits=k, shuffle=True,random_state=42)

# Perform k-fold cross-validation
for i, (train_index,test_index) in enumerate(kf.split(X)):
    # split the data into training and test sets
    training_set,test_set=X[train_index],X[test_index]
    training_labels,test_labels=Y[train_index],Y[test_index]
    dt_classifier=DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(training_set,trainig_labels)
    
    #make predictions on the validation set
    fold_predictions=dt_classifier.predict(test_set)
    
    #calculate the accuracy of the fold
    fold_accuracy= np.mean(fold_predictions==test_labels)
    
    #store the predictions and accuracy of each fold
    predictions[i]=fold_predictions
    accuracies.append(fold_accuracy)
    
    #print the predictions and accuracy of each fold
    print("Fold{}:Accuracy:{:.4f}".format(i+1,fold_accuracy))
# overall accuracy across all folds
print(f"Overall Accuracy:{np.mean(accuracies):.4f}")

# nested cross_validation to find the optical depth
depths=[1,2,3,4,5,6,7,8,9,10,None]
nested_accuracies=[]
for depth in depths:
    kf= KFold(n_splits=k,shuffle=True,random_state=42)
    fold_accuracies=[]
    for i, (train_index,test_index) in enumerate(kf.split(X)):
        #split the data into training and test sets
        training_set,test_set=X[train_index],X[test_index]
        training_labels,test_labels=Y[train_index],Y[test_index]

        #Trian the model with the current depth
        dt_classifier= DecisionTreeClassifier(max_depth=depth,random_state=42)
        dt_classifier.fit(training_set,trainig_labels)

        #make predictions on the validation set
        fold_predictions=dt.classifier.predict(test_set)

        #calculate the accuracy of the fold
        fold_accuracy=np.mean(fold_predictions==test_labels)
        fold_accuracies.append(fold_accuracy)

    # store the mean accuracy for the depth
    nested_accuracies.append(np.mean(fold_accuracies))
    print(f"Depth{depth}:Avg Accuracy:{np.mean(fold_accuracies):.4f}")

# Determine the depth with the highest average accuracy 
optimal_depth= depths[np.argmax(nested_accuracies)]
print(f"Optimal Depth:{optimal_depth},Best average Accuracy:{max(nested_accuracies):.4f}")





    



