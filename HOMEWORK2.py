"""
HOMEWORK 2 GABRIEL LOPES up201607742 [MIB]-DACO

"""
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree
from sklearn import metrics



data = load_breast_cancer(return_X_y = True)# load the databases
features = data [0]
targets = data [1]
print (features.shape, targets.shape)

print ('Positive Samples Proportion', np.sum(targets ==1)/np.size(targets))

#features1 = StandardScaler().fit_transform(features) #To increase performance but in this case I think inappropriate
X_train, X_test, y_train, y_test = train_test_split(features,targets,test_size=0.20,random_state=10)

#####################################################################################################################

#Naive Bayes (no hyperparameter tuning)
print('\n########################################-NAIVE BAYES-########################################\n')
gnb = GaussianNB()
scores = cross_val_score(gnb, X_train, y_train, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#####################################################################################################################

#Logistic Regression
print('\n########################################-LOGISTIC REGRESSION-########################################\n')
# create a train/val division

regularization_params = [0.0001, 0.001, 0.01, 1, 10]

cv_scores_array=[]
for C in regularization_params:
    print('Evaluating parameter', C)
    clf_LR = LogisticRegression(C=C)
    cv_scores = cross_val_score(clf_LR, X_train, y_train, cv=10)
    print('Mean accuracy across folds:', np.mean(cv_scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    cv_scores_array.append(cv_scores.mean())



optimal_regularization =regularization_params[np.argmax(cv_scores_array)]
clf_LR = LogisticRegression(C=optimal_regularization)
clf_LR.fit(X_train, y_train)
clf_LR.score(X_test, y_test)
print("Accuracy: " , (clf_LR.score(X_test,y_test)))

#####################################################################################################################

#k-Nearest Neighbours
print('\n########################################-K-NEAREST NEIGHBOURS-########################################\n')

n_neighbors = [1,2,3,4,5,6,7,8,9,10]
bestnumber= []
for n in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn_scores = cross_val_score(knn, X_train, y_train, cv=10)
    print('the neighbors mean Accuracy:', np.mean(knn_scores))
    print('Accuracy:  %0.2f (+/- %0.2f)' % (knn_scores.mean(), knn_scores.std() * 2))
    bestnumber.append(knn_scores.mean())

optimal_number_neighbors = n_neighbors[np.argmax(bestnumber)]
knn = KNeighborsClassifier(n_neighbors = optimal_number_neighbors)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
print("Accuracy: " , (knn.score(X_test,y_test)))



#####################################################################################################################

#Support Vector Machines
print('\n########################################-SVM-########################################\n')

# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [0.01, 0.1],
                'C': [1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=3)

print("# Tuning hyper-parameters")
clf.fit(X_train[:10,:], y_train[:10]);

print("Best parameters set found on validation set: \n")
print(clf.best_params_)
print("Grid scores on training set: \n")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/- %0.03f) for %r" % (mean, std * 2, params))

clf_x = SVC(probability =True, **clf.best_params_) # need to use ** for put the parameters
clf_x.fit(X_train, y_train)
SVM_Accuracy= clf_x.score(X_test, y_test)
print ("Accuracy of SVM is: ", SVM_Accuracy)


#####################################################################################################################      

#Decision Trees
print('\n########################################-DECISION TREES-########################################\n')
parameters  = {"criterion": ["gini", "entropy"],
               "min_samples_split": [2, 10, 20],
               "max_depth": [None, 2, 5, 10],
               "min_samples_leaf": [1, 5, 10],
               "max_leaf_nodes": [None, 5, 10, 20],
        }

clf_tree = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=10)

print("# Tuning hyper-parameters")
clf_tree.fit(X_train[:20,:], y_train[:20]);

print("Best parameters set found on validation set: \n")
print(clf_tree.best_params_)
print("Grid scores on training set: \n")
means = clf_tree.cv_results_['mean_test_score']
stds = clf_tree.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf_tree.cv_results_['params']):
    print("%0.3f (+/- %0.03f) for %r" % (mean, std * 2, params))

clf_tree_x = tree.DecisionTreeClassifier()
clf_tree_x.fit(X_train, y_train)
TREE_Accuracy= clf_tree_x.score(X_test, y_test)  
print ("Accuracy of Tree is: ", TREE_Accuracy)



######################################################################################################################
#Calculate TP, TN, FP, FN


y_pred = knn.predict(X_test)

TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()
print(TN, FP, FN, TP)

#Sensitivity
sensitivity = TP / float(FN + TP)
print("The Sensitivity is: ",sensitivity)

#specificity
specificity = TN / (TN + FP)
print("The Specificity is: ",specificity)

#Precision
precision = TP / float(TP + FP)
print("The Precision is: ", precision)


knn.predict(X_test)[0:10]
knn.predict_proba(X_test)[0:10]

y_pred_prob = knn.predict_proba(X_test)[:, 1]


 # Histogram
plt.hist(y_pred_prob, bins=8)
# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities');
plt.xlabel('Predicted probability of diabetes');
plt.ylabel('Frequency');

#ROC-Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

def evaluate_threshold(threshold):
 print('Sensitivity:', tpr[thresholds > threshold][-1])
 print('Specificity:', 1 - fpr[thresholds > threshold][-1])
 print('threshold =', 0.5)
 evaluate_threshold(0.5)
 print('------')
 print('threshold =', 0.3)
 evaluate_threshold(0.3)