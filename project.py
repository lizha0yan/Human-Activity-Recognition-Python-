# Decision Tree Classification

# Importing the libraries
import pandas as pd

# Importing the training dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:562].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and validation set
from sklearn.cross_validation import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 7)
classifier.fit(X_train, y_train)

# Predicting the training and validation set results
y_pred_train = classifier.predict(X_train)
y_pred_validation = classifier.predict(X_validation)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_pred_train)
#training error=0
cm_validation = confusion_matrix(y_validation, y_pred_validation)
#validation error=0.05005

# Visualising the Decision Tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=["1","2","3","4","5","6"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#Testing the testing error

#Importing the testing dataset 
testset = pd.read_csv('test.csv')
X_test = testset.iloc[:,1:562].values
y_test = testset.iloc[:,0].values

# Predicting the testing set results
y_pred_test = classifier.predict(X_test)

#Making the Confusion Matrix
cm_test = confusion_matrix(y_test, y_pred_test)
#testing error=0.1751

#Using Random Forest model to impove accuracy 
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 700,criterion = 'entropy', random_state= 7)
classifier_rf.fit(X_train, y_train)

# Predicting the training and validation set results
y_pred_train_rf = classifier_rf.predict(X_train)
y_pred_validation_rf = classifier_rf.predict(X_validation)

# Making the Confusion Matrix
cm_train_rf = confusion_matrix(y_train, y_pred_train_rf)
#training error = 0
cm_validation_rf = confusion_matrix(y_validation, y_pred_validation_rf)
#validation error = 0.0185

# Predicting the testing set results
y_pred_test_rf = classifier_rf.predict(X_test)

#Making the Confusion Matrix
cm_test_rf = confusion_matrix(y_test, y_pred_test_rf)
#testing error=0.0719


