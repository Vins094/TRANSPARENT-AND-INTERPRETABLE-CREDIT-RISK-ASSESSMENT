import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as  plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score
from joblib import load
from sklearn.metrics import RocCurveDisplay
import os
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import torch
from skorch import NeuralNetClassifier
import torch.nn as nn
import torch.nn.functional as F
import os
# Google Drive path where project material is available
GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Dissertation'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))


#load data function
def train_test_split_func(random_state):
  X = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH, 'x_data.csv'))
  y = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH, 'y_data.csv'))
  x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y, random_state = random_state)
  
  return x_train, x_test, y_train, y_test

#load Decsion tree model and print its performance and tree
def dt_model_performance(dt_model, train_test_split_func):
  x_train, x_test, y_train, y_test = train_test_split_func(129)
  y_pred = dt_model.predict(x_test)
  print("Classification Report:") #print classification report
  print(classification_report(y_test, y_pred))
  # predictions to calculate performance metrices
  y_train_pred_dt = dt_model.predict(x_train)
  y_test_pred_dt= dt_model.predict(x_test)
  y_test_pred_proba_dt = dt_model.predict_proba(x_test)[:, 1]

  # performance metrices
  train_accuracy = accuracy_score(y_train, y_train_pred_dt)
  test_accuracy = accuracy_score(y_test, y_test_pred_dt)
  precision = precision_score(y_test, y_test_pred_dt)
  recall = recall_score(y_test, y_test_pred_dt)
  f1 = f1_score(y_test, y_test_pred_dt)
  auc = roc_auc_score(y_test, y_test_pred_proba_dt)

  # Print performance results
  print(f"Train Accuracy of decsion tree: {train_accuracy*100}")
  print(f"Test Accuracy of decsion tree: {test_accuracy*100}")
  print(f"Precision decsion tree: {precision*100}")
  print(f"Recall decsion tree: {recall *100}")
  print(f"F1 Score decsion tree: {f1*100}")
  print(f"AUC decsion tree: {auc *100}")

  #display confusion matrics
  cm_dt = confusion_matrix(y_test, y_pred)
  display_cm= ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Fully paid', 'Charged Off'])
  display_cm.plot(cmap='Blues', values_format='d')
  plt.title('Confusion Matrix')
  plt.show()

  plt.figure(figsize = (20,10))
  tree.plot_tree(dt_model, feature_names= list(x_test.columns), class_names= ['Fully Paid', 'Default'], filled = True)


#logistic regression model performance and explainability function
def logistic_regression_performance(lr_model,train_test_split_func):
  x_train, x_test, y_train, y_test = train_test_split_func(129)
  pred= lr_model.predict(x_test) 
  print("Classification Report of Logistic Regression:")
  print(classification_report(y_test, pred))
  cm_lr = confusion_matrix(y_test, pred)
  display_cm = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Fully paid', 'Charged Off'])
  display_cm.plot(cmap='Blues', values_format='d')
  plt.title('Confusion Matrix of Logistic Regression')
  plt.show()
  # predictions to calculate performance metrices
  y_train_pred_lr = lr_model.predict(x_train)
  y_test_pred_lr = lr_model.predict(x_test)
  y_test_pred_proba_lr = lr_model.predict_proba(x_test)[:, 1]

  # performance metrices
  train_accuracy = accuracy_score(y_train, y_train_pred_lr)
  test_accuracy = accuracy_score(y_test, y_test_pred_lr)
  precision = precision_score(y_test, y_test_pred_lr)
  recall = recall_score(y_test, y_test_pred_lr)
  f1 = f1_score(y_test, y_test_pred_lr)
  auc = roc_auc_score(y_test, y_test_pred_proba_lr)

  # Print performance results
  print(f"Train Accuracy of Logistic Regression: {train_accuracy*100}")
  print(f"Test Accuracy of Logistic Regression: {test_accuracy*100}")
  print(f"Precision of Logistic Regression: {precision*100}")
  print(f"Recall of Logistic Regression: {recall *100}")
  print(f"F1 Score of Logistic Regression: {f1*100}")
  print(f"AUC of Logistic Regression: {auc *100}")

  X = sm.add_constant(x_train)  # add constant to the independent variable
  lr_model_ex = sm.Logit(y_train, X).fit() #explain model using stats model on default values, as it is directly not possible to achive coeficient and marginal effect so using default parameters for explanations
  #print model summary
  print(lr_model_ex.summary())
  # print the odd ratios
  odds_ratios_lr = np.exp(lr_model_ex.params)
  print("Odd Ratios are :")
  print(odds_ratios_lr)
  # Calculate average marginal effects
  marginal_effects = lr_model_ex.get_margeff()
  print(f'Marignal effects of Default Logistic regression model are: {marginal_effects.summary()}')


#Random forest performance function
def rf_performance(rf_model,train_test_split_func):
  x_train, x_test, y_train, y_test = train_test_split_func(129)
  pred = rf_model.predict(x_test)
  cm_5 = confusion_matrix(y_test, pred)
  disp_cm5 = ConfusionMatrixDisplay(confusion_matrix=cm_5, display_labels=['Fully paid', 'Charged Off'])
  disp_cm5.plot(cmap='Blues', values_format='d')
  plt.title('Confusion Matrix')
  plt.show()
  # predictions to calculate performance metrices
  y_train_pred_rf = rf_model.predict(x_train)
  y_test_pred_rf= rf_model.predict(x_test)
  y_test_pred_proba_rf = rf_model.predict_proba(x_test)[:, 1]

  # performance metrices
  train_accuracy = accuracy_score(y_train, y_train_pred_rf)
  test_accuracy = accuracy_score(y_test, y_test_pred_rf)
  precision = precision_score(y_test, y_test_pred_rf)
  recall = recall_score(y_test, y_test_pred_rf)
  f1 = f1_score(y_test, y_test_pred_rf)
  auc = roc_auc_score(y_test, y_test_pred_proba_rf)

  # Print performance results
  print(f"Train Accuracy of Random forest: {train_accuracy*100}")
  print(f"Test Accuracy of Random forest: {test_accuracy*100}")
  print(f"Precision of Random forest: {precision*100}")
  print(f"Recall of Random forest: {recall *100}")
  print(f"F1 Score Random forest: {f1*100}")
  print(f"AUC Random forest: {auc *100}")

  feature_importance =rf_model.feature_importances_
  feature_importance = pd.Series(feature_importance, index = x_train.columns).sort_values(ascending=False)
  feature_importance.plot(kind = 'bar', figsize=(30,5))
  plt.ylabel("Feature Importance")


#XGBoost model performance
def xgb_performance(xgb_model,train_test_split_func):
  x_train, x_test, y_train, y_test = train_test_split_func(7)
  x_train1 = x_train.values
  x_test1 = x_test.values
  y_train1 = y_train.values
  y_test1 = y_test.values
  # Make predictions
  y_train_pred_xg = xgb_model.predict(x_train1)
  y_test_pred_xg = xgb_model.predict(x_test1)
  y_test_pred_proba_xg = xgb_model.predict_proba(x_test1)[:, 1]

  # # performance metrices
  train_accuracy = accuracy_score(y_train1, y_train_pred_xg)
  test_accuracy = accuracy_score(y_test1, y_test_pred_xg)
  precision = precision_score(y_test1, y_test_pred_xg)
  recall = recall_score(y_test1, y_test_pred_xg)
  f1 = f1_score(y_test1, y_test_pred_xg)
  auc = roc_auc_score(y_test1, y_test_pred_proba_xg)

  # Print performance results
  print(f"Train Accuracy of xgBoost: {train_accuracy*100}")
  print(f"Test Accuracy of xgBoost: {test_accuracy*100}")
  print(f"Precision of xgBoost: {precision*100}")
  print(f"Recall of xgBoost: {recall *100}")
  print(f"F1 Score of xgBoost: {f1*100}")
  print(f"AUC of xgBoost: {auc *100}")

  cm_xg= confusion_matrix(y_test1, y_test_pred_xg)
  display_cm = ConfusionMatrixDisplay(confusion_matrix=cm_xg, display_labels=['Fully paid', 'Charged Off'])
  display_cm.plot(cmap='Blues', values_format='d')
  plt.title('Confusion Matrix')
  plt.show()

#ann network defined
class credit_net(nn.Module):
    def __init__(self,
                 input_dim = 20,
                 hidden_dim = [100, 120, 50] ,
                 output_dim = 2,
                 dropout = 0.4
                ):
        super(credit_net, self).__init__() #instantiate nn.module
        self.fc1= nn.Linear(input_layer, hidden_layers[0]) #20 features will have 20 neurons in input layer
        self.dropout = nn.Dropout(dropout)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])

        self.out = nn.Linear(hidden_dim[-1] ,output_layer)
        self.relu = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim =-1)





    def forward(self, x): #x is input
        x = self.relu(self.fc1(x)) #activation function is firing here
        x = self.dropout(x)#droput
        for layer in self.hidden_layers: # to reduce the code we have used for loop to define hidden layer instead of explicily define each layer
            x = self.LeakyReLU(layer(x))
            x = self.dropout(x)

        x = self.softmax(self.out(x)) #softmax function firing on output layer, it will give probability of each class(gamma or Hadron)


        return x

#neural network function and its performance
def ann_performance(ann_model,train_test_split_func):
  x_train, x_test, y_train, y_test = train_test_split_func(129)
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  y_train = np.array(y_train).ravel()
  y_test = np.array(y_test).ravel()
  x_train = x_train.astype(np.float32) #datatype required by skorch for model training
  y_train = y_train.astype(np.int64)
  x_test = x_test.astype(np.float32)
  y_test = y_test.astype(np.int64)
  ann_model.load_params
  y_pred = ann_model.predict(x_test)
  # predictions to calculate performance metrices
  y_train_pred_mlp = ann_model.predict(x_train)
  y_test_pred_mlp = ann_model.predict(x_test)
  y_test_pred_proba_mlp = ann_model.predict_proba(x_test)[:, 1]

  # performance metrices
  train_accuracy = accuracy_score(y_train, y_train_pred_mlp)
  test_accuracy = accuracy_score(y_test, y_test_pred_mlp)
  precision = precision_score(y_test, y_test_pred_mlp)
  recall = recall_score(y_test, y_test_pred_mlp)
  f1 = f1_score(y_test, y_test_pred_mlp)
  auc = roc_auc_score(y_test, y_test_pred_proba_mlp)

  # Print performance results
  print(f"Train Accuracy of mlp: {train_accuracy*100}")
  print(f"Test Accuracy of mlp: {test_accuracy*100}")
  print(f"Precision of mlp: {precision*100}")
  print(f"Recall mlp: {recall *100}")
  print(f"F1 Score of mlp {f1*100}")
  print(f"AUC of mlp: {auc *100}")

  cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
  disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
  disp_mlp.plot(cmap='Blues', values_format='d')
  plt.title('Confusion Matrix')
  plt.show()

  


