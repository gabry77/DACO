# HomeWork By Gabriel Lopes up201607742 MIB-[DACO]

import pandas as pd
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', header=None)
df.columns = ['#pregnant', 'glucose', 'blood_pressure', 'thickness', 'insulin', 'body_mass_idx', 'pedigree', 'age', 'label']

df.head()

dataset = np.array(df)
features = dataset[:, :8]
labels = dataset[:, -1]

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

########################################################################################################
#Naive Bayes classifier
gnb = GaussianNB()
Train_Model = gnb.fit(X_train, y_train)
Test_Model=gnb.predict(X_test)


# Print Predictions and Label Function
def print_predictions_and_labels(array_preds_labels):
    for predicted_pair in array_preds_labels:
        prediction = predicted_pair[0]
        label = predicted_pair[1]
        print('Prediction', prediction, 'Label', label)
        
#Store values in array with predictions in frist column
Naive_Bayes_preds=np.concatenate([Test_Model.reshape(-1,1), y_test.reshape(-1,1)], axis=1)
        
#print(Naive_Bayes_preds)
print_predictions_and_labels (Naive_Bayes_preds)   

#Accuracy measurment 1  
print('The Accuracy is ', accuracy_score(y_test, Test_Model, normalize=False)/len(y_test))
#print('The accurary is ', accuracy_score(y_test, Test_Model, normalize=True))
#we can use both 

#Acuraçy measurment 2
correctAnswers=(Test_Model==y_test)
accuracy=np.count_nonzero(correctAnswers)/(Test_Model.size)
print('The Accuracy is:', accuracy)

#Acuraçy measurment 2
print ('The Accuracy Score:', Train_Model.score(X_test, y_test))





########################################################################################################
#Linear Regression

diabetes_dataset = datasets.load_diabetes(return_X_y=True)

features=diabetes_dataset[0]
labels=diabetes_dataset[1]

#Create training and testing vars with 30%
X_train2, X_test2, y_train2, y_test2 = train_test_split(features, labels, test_size=0.3, random_state=0)

# fit a model
lm = linear_model.LinearRegression()
lm_model = lm.fit(X_train2, y_train2)
predictions =lm.predict(X_test2)
plt.scatter(y_test2, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

#MSE error
MSE_error=mean_squared_error(y_test2, predictions)
print('Mean Squared Error is: \n',MSE_error)

###########################################################################################################
#6.3 Ungraded Homework - k neighboors
#k-nearest neighbor classification
iris = datasets.load_iris()

subset = np.logical_or(iris.target == 0, iris.target == 1)

k=3 
X1 = iris.data[subset]
Y1 = iris.target[subset]

def distance(x,y):
    return np.linalg.norm(x-y)


def knntrain(Xknn_train, yknn_train):
    return

#Creating the predict block of KNN Classifier

def predictknn(Xknn_train, yknn_train, Xknn_test, k1):
    knndistances = np.zeros((Xknn_train.shape[0],2))
    #knntargets = []
    
    for i in range(Xknn_train.shape[0]):
        #Compute Euclidean Distance, we can compute like d, or use the function above
        # d = np.sqrt(np.sum(np.square(Xknn_test - Xknn_train[i, :])))
        knndistances[i][0]=distance(Xknn_test, Xknn_train[i,:]) 
        #Add it to list of distances
        knndistances[i][1]=i;
    
    #Sort the list, we can compute with the function sorted
    #knndistances = sorted(knndistances)
    Indexes=knndistances[:,0].argsort();
    CorrespondingLabels=yknn_train[Indexes[0:k]];
    
    #Get and return most common knntarget
    return Counter(CorrespondingLabels).most_common(1)[0][0]

###########################################################################################################
#for trainning
xnew = np.array([3.5, 2.5, 2.5, 0.75])

predictknn(X1, Y1, xnew, k)   
###########################################################################################################
#With a test split
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X1, Y1, test_size=0.30, random_state=42)

Predictions = []
for l in range (X_knn_test.shape[0]):
    Predictions.append(predictknn(X1, Y1, X_knn_test[l,:], k))

kpreds = np.array(Predictions)

#Print predictions
print("Predicted values are: \n", kpreds)

#Print accuracy 
correctAnswers=(kpreds==y_knn_test)
accuracy=np.count_nonzero(correctAnswers)/(len(kpreds))
print("The accuracy from the tradicional method is: \n",accuracy)

    
###########################################################################################################
#With sklearn library

knn = KNeighborsClassifier(k)

#Fitting the model
knn.fit(X_knn_train, y_knn_train)

#Predict the response
skl_pred = knn.predict(X_knn_test)
print("Predicted values are: \n", skl_pred)

#Evaluate accuracy
print("Accuracy for knn is: \n", accuracy_score(skl_pred, kpreds))