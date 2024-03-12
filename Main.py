from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import END


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE


from keras.callbacks import ModelCheckpoint 
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam

main = tkinter.Tk()
main.title("Spacecraft Decision Making using DL Rule Master Generated Rules") #designing main screen
main.geometry("1300x1200")

global filename, dataset
global X, Y
global X_train, X_test, y_train, y_test, X_train_smote, y_train_smote
global accuracy, precision, recall, fscore, labels, nin_model
global scaler, labels, label_encoder
accuracy = []
precision = []
recall = []
fscore = []

def uploadDataset():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset= pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels, label_count = np.unique(dataset['Decision'], return_counts=True)
    #label = dataset.groupby('Decision').size()
    #label.plot(kind="bar")
    plt.show()
    sns.countplot(x = dataset['Decision'])
    plt.xlabel("Decision Category Type")
    plt.ylabel("Count")
    plt.title("Decision Category Graph")
    plt.show()

def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    dataset = pd.read_csv(filename)
    global X_train, X_test, y_train, y_test, scaler, X_train_smote, y_train_smote
    print(dataset.describe())
    print(dataset.info())
    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END,str(dataset)+"\n\n")
    X = dataset.drop(['Decision'],axis = 1) 
    Y = dataset['Decision']
    Y
    print(X.shape)
    print(np.unique(Y))
    print(Y)
    
    scaler = StandardScaler()
    r = ['Solar Radiation', 'Temperature', 'Battery Level', 'Engine Thrust', 'Oxygen Level']
    for i in r:
        X[i] =scaler.fit_transform (X[[i]])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) 
    print(X_train.shape)
    #smote = SMOTE(random_state=42)
    #X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    #sns.countplot(x = y_train_smote)
    #plt.show()

    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global labels, X_train_smote, y_train_smote
    #global accuracy = [], precision = [], recall = [], fscore = []
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

#now train existing standard CNN algorithm    

def runStandardCNN():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test, X_train_smote, y_train_smote,model_mlp
    accuracy = []
    precision = []
    recall = [] 
    fscore = []

    model_mlp = Sequential()

    model_mlp.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model_mlp.add(Dense(32, activation='relu'))
    model_mlp.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model_mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_mlp.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    loss_mlp, accuracy_mlp = model_mlp.evaluate(X_test, y_test)
    print(f"MLP Model - Test Accuracy: {accuracy_mlp * 100:.2f}%")
    y_pred_mlp = model_mlp.predict(X_test)
    # Convert predicted probabilities to binary labels
    y_pred_mlp = (y_pred_mlp > 0.5).astype(int)

    predict = y_pred_mlp
    testY = y_test
    calculateMetrics("Residual Neural Networks Model", testY, predict)
    #calculateMetrics("Existing Standard Logistic Regression ", testY, pred)



def runLogisticRegression():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore,clf
    global X_train, y_train, X_test, y_test, X_train_smote, y_train_smote
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    #pred = np.argmax(pred, axis=1)
    #testY = np.argmax(y_test, axis=1)
    testY = y_test
    calculateMetrics("Logistic Regression ", testY, pred)
   

def graph():
    df = pd.DataFrame({
        'Algorithms': ['Residual Neural Networks', 'Residual Neural Networks', 'Residual Neural Networks', 'Residual Neural Networks', 'Logistic Regression', 'Logistic Regression', 'Logistic Regression', 'Logistic Regression'],
        'Accuracy': ['Accuracy', 'Precision', 'Recall', 'FSCORE', 'Accuracy', 'Precision', 'Recall', 'FSCORE'],
        'Value': [accuracy[0], precision[0], recall[0], fscore[0], accuracy[1], precision[1], recall[1], fscore[1]]
    })

    # Convert 'Value' column to numeric for correct plotting
    df['Value'] = pd.to_numeric(df['Value'])

    # Pivot the DataFrame for plotting
    df_pivot = df.pivot(index='Algorithms', columns='Accuracy', values='Value')

    # Plotting
    ax = df_pivot.plot(kind='bar')
    ax.set_xticklabels(df_pivot.index, rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.title("All Algorithm Comparison Graph")
    plt.show()
    
def predict():
    global model_mlp, scaler, labels
    text.delete('1.0', END)
    global X_test
    print(X_test)
    
    for i in range(len(X_test)):
        input_data = X_test.iloc[i, :].values.reshape(1, -1)
        prediction = model_mlp.predict(input_data)
        prediction = (prediction > 0.5).astype(int)
        text.insert(END, f'Input data for row {i}: {input_data}\n')
        
        if prediction[0] == 0:
            predicted_data = "0"
        elif prediction[0] == 1:
            predicted_data = "1"
         
        text.insert(END, f'Predicted output for row {i}: {predicted_data}\n')

font = ('times', 16, 'bold')
title = Label(main, text='Spacecraft Decision Making using DL Rule Master Generated Rules')
title.config(bg='LightGreen', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Dataset Preprocessing", command=DatasetPreprocessing)
preButton.place(x=370,y=100)
preButton.config(font=font1) 

rfButton = Button(main, text="Logistic Regression", command=runLogisticRegression)
rfButton.place(x=860,y=100)
rfButton.config(font=font1)

nbButton = Button(main, text="Residual Neural Networks", command=runStandardCNN)
nbButton.place(x=610,y=100)
nbButton.config(font=font1) 
 

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Prediction on Test Data", command=predict)
predictButton.place(x=370,y=150)
predictButton.config(font=font1)  

#main.config(bg='OliveDrab2')
main.mainloop()
