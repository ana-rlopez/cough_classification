"""
This module includes functions related with the training of the classification model, and class prediction based on the trained model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def modelTraining(X_train,y_train,ID_train):
    """
    Train a logistic regression classifier, and use one-leave-out approach to do crossvalidation.
    
    Args:
        X_train (dataframe of features, once removed the columns of label and recording indexes): input train data.
        y_train (dataframe with just column of labels): output train data.
        ID_train (dataframe with just column of IDs of the training set): IDs of the audio recordings.   
        
    Returns:
        pred_probs (dataframe): prediction probabilities (one column per class).
    """
    #TODO convert the function such that it can takes a general classifier (class).
    ID_list = ID_train.drop_duplicates().tolist()
    pred_probs = pd.DataFrame([])

    #leave-one-out based on recordings (we leave out one recording as validation set at a time)
    for i in range(0,len(ID_list)):

        idnow = ID_list[i]
        ID_train_list = ID_train.to_list()
        val_index = [i for i, x in enumerate(ID_train_list) if x == idnow]
        train_index = [i for i, x in enumerate(ID_train_list) if x != idnow]

        X_train1, X_val1 = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train1, y_val1 = y_train.iloc[train_index], y_train.iloc[val_index]

        #normalize train set
        scaler = StandardScaler()
        scaler.fit(X_train1)
        X_trainNorm1 = scaler.transform(X_train1.values)
        X_valNorm1 = scaler.transform(X_val1.values)

        #TODO: optimize the penaly weight
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        logReg = SGDClassifier(loss='log', penalty='elasticnet')
        logReg.fit(X_trainNorm1, y_train1)
        y_hat_prob = logReg.predict_proba(X_valNorm1)
        classes =logReg.classes_
        pred_probs = pred_probs.append(pd.DataFrame({'ID': ID_train[val_index], str(classes[0]): y_hat_prob[:,0],
                                                     str(classes[1]): y_hat_prob[:,1]}),ignore_index=True, sort=False)    
    return pred_probs

def predict_class(prob_dry,prob_wet):
    """
    Predict class ('Wet' or 'Dry') based on the input prediction probabilities for each class.
    
    Args:
        prod_dry (dataframe): prediction probabilities for class 'Dry'.
        prod_wet (dataframe): prediction probabilities for class 'Wet'.
        
    Returns:
        'Dry'/'Wet' (string): predicted class.
    """
    if prob_dry > prob_wet :
        return 'Dry'
    else:
        return 'Wet'

def get_predClass_per_audio(pred_probs, label_dict):
    """    
    Get probability per recording, by averaging the probabilities for each class over all entries (at chunk-level) \
    of a recording.
    
    Args:
        pred_probs (dataframe): dataframe containing the predicted probabilities of each class in a column (chunk-level).
        label_dict (dictionary): dictionary of label vs ID of a recording.

    Returns:
        pred_probs_perID (dataframe): one predicted probability per class (in a column), but now per ID (i.e. audio recording) \
        rather than per chunk.
    """

    pred_probs_perID = pred_probs.groupby('ID').aggregate('mean').reset_index()

    pred_probs_perID['pred_class'] = pred_probs_perID.apply(lambda x: predict_class(x['Dry'], x['Wet']), axis=1)

    #add actual classes
    pred_probs_perID['label'] = pred_probs_perID["ID"].map(label_dict)
    return pred_probs_perID
