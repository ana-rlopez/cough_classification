"""
This module includes functions related with the evaluation of the classification task.
"""

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import pandas as pd

import modelTrainingFunctions as modelTrainLib

def compute_classificationScores(true_classes, predicted_classes):
    """
    Compute classification scores (accuracy, precision, f1 and recall) from the predicted and true classes.
    
    Args:
        true_classes (list of int/binaries): the true labels/classes.
        predicted_classes (list of int/binaries): the predicted labels/classes using the trained classifier.
    
    Returns:
        scores (dictionary): dictionary of the computed scores (accuracy, precision, f1 and recall).
    """
    scores = {}
    scores['accuracy'] = accuracy_score(true_classes, predicted_classes)
    scores['precision'] = precision_score(true_classes, predicted_classes,average="macro")
    scores['f1'] = f1_score(true_classes, predicted_classes,average="macro")
    scores['recall'] = recall_score(true_classes, predicted_classes,average="macro")
    
    #TODO: add sensitivity and specificity scores
    #TODO include confusion matrix
    #confusion matrix
    #conf_mat_df = pd.crosstab(mean_pred_probs['label'], mean_pred_probs['pred_class'], margins=True)
    
    return scores
def print_scores(scores):
    """
    Print the input scores.
        
    Args:
        scores (dictionary/series): dictionary or pandas series with the scores to print.
    
    Returns:
        None
    """
    
    if isinstance(scores,dict):
        #convert dictionary to pandas series
        scores = pd.Series(scores, index =scores.keys()) 
    
    print(scores.to_string())

    return

def getScores_perTrainRun(X_train,y_train,ID_train,label_dict, scores_list):
    """
    Get the specified scores from one run of the training (and prediction) of the model (with one-leave-out crossvalidation).
    
    Args:
        X_train (dataframe of features, once removed the columns of label and recording indexes): input train data.
        y_train (dataframe with just column of labels): output train data.
        ID_train (dataframe with just column of IDs of the training set): IDs of the audio recordings.
        label_dict (dictionary): dictionary of label vs ID of a recording.
        score_list (list of strings): list with the scores of interest. \
        Possible ones:'accuracy', 'precision', 'f1' and 'recall'.
    
    Returns:
        select_scores (dictionary): dictionary containing all the scores of the specified score list (for 1 run).
    """

    pred_probs = modelTrainLib.modelTraining(X_train,y_train,ID_train)
    pred_probs_perID = modelTrainLib.get_predClass_per_audio(pred_probs, label_dict)
      
    scores = compute_classificationScores(pred_probs_perID['label'], pred_probs_perID['pred_class'])       
    
    select_scores = { select_key: scores[select_key] for select_key in scores_list }
    
    return select_scores

def getAvgEvalScores_ofTrainRuns(nr_runs,X_train,y_train,ID_train,label_dict, scores_list, print_flag):
    """
    Get the average scores of the specified score list from a number of runs.
    
    Args:
        nr_runs (int): number of runs.
        X_train (dataframe of features, once removed the columns of label and recording indexes): input train data.
        y_train (dataframe with just column of labels): output train data.
        ID_train (dataframe with just column of IDs of the training set): IDs of the audio recordings.
        label_dict (dictionary): dictionary of label vs ID of a recording.
        score_list (list of strings): list with the scores of interest. \
        Possible ones:'accuracy', 'precision', 'f1' and 'recall'.
        print_flag (True/False): flag to indicate if printing of scores per run is done or not.
    
    Returns:
        avg_scores (series): pandas series containing all the average scores of the specified score list.
    """    
    #initialize dataframe to store scores over runs
    allScores_df = pd.DataFrame(columns = scores_list)
    
    for i in range(nr_runs):

        select_scores = getScores_perTrainRun(X_train,y_train,ID_train,label_dict, scores_list)
          
        if print_flag is True:            
            print('Run %d:\n' % i)
            print_scores(select_scores)
            print('\n')
        
        new_score = pd.DataFrame({key: [value] for key, value in select_scores.items()})
        allScores_df = allScores_df.append(new_score, ignore_index=True)
       
    avg_scores = allScores_df.mean()
    return avg_scores