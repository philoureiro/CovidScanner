import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

 
def print_menu():      
    print(30 * '-' , 'MENU' , 30 * '-')
    print('1. Gray Histogram')
    print('2. Sift')
    print('3. Haralick')
    print('4. HuMoments')
    print('5. Exit')
    print(67 * '-')

def menu_choice():
    print_menu()
    featureNames = ['grayhistogram','sift','haralick','humoments']
    choice = int(input('Escolha o número da opção desejada: '))
    return featureNames[choice-1]
    
def main():
    featureName = menu_choice()
    trainFeaturePath = f'./features_labels/{featureName}/train/'
    testFeaturePath = f'./features_labels/{featureName}/test/'
    featureFilename = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoderClasses.csv'
    mainStartTime = time.time()
    print(f'[INFO] ========= TRAINING PHASE ========= ')
    trainFeatures = getFeatures(trainFeaturePath,featureFilename)
    trainEncodedLabels = getLabels(trainFeaturePath,labelFilename)
    sgd = trainSGD(trainFeatures,trainEncodedLabels)
    print(f'[INFO] =========== TEST PHASE =========== ')
    testFeatures = getFeatures(testFeaturePath,featureFilename)
    testEncodedLabels = getLabels(testFeaturePath,labelFilename)
    encoderClasses = getEncoderClasses(testFeaturePath,encoderFilename)
    predictedLabels = predictSGD(sgd,testFeatures)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Code execution time: {elapsedTime}s')
    accuracy = plotConfusionMatrix(encoderClasses,testEncodedLabels,predictedLabels,featureName)
    return accuracy, featureName

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        return folder_path

def getFeatures(path,filename):
    features = np.loadtxt(path+filename, delimiter=',')
    return features

def getLabels(path,filename):
    encodedLabels = np.loadtxt(path+filename, delimiter=',',dtype=int)
    return encodedLabels

def getEncoderClasses(path,filename):
    encoderClasses = np.loadtxt(path+filename, delimiter=',',dtype=str)
    return encoderClasses

def trainSGD(trainData,trainLabels):
    print('[INFO] Training the SGD model...')
    sgd_pipeline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    startTime = time.time()
    sgd_pipeline.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return sgd_pipeline

def predictSGD(sgd_pipeline,testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels =  sgd_pipeline.predict(testData)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels

def getCurrentFileNameAndDateTime():
    fileName =  os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime

def plotConfusionMatrix(encoderClasses,testEncodedLabels,predictedLabels,featureName):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    #Decoding test labels from numerical labels to string labels
    test = encoder.inverse_transform(testEncodedLabels)
    #Decoding predicted labels from numerical labels to string labels
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test,pred,ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: '+featureName+'-'+getCurrentFileNameAndDateTime(),fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels,predictedLabels)*100
    plt.title(f'Accuracy: {accuracy}%',fontsize=18,weight='bold')
    plt.savefig('./results/'+featureName+'-'+getCurrentFileNameAndDateTime(), dpi=300)  
    print(f'[INFO] Plotting done!')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)
    return accuracy

if __name__ == "__main__":
    main()
