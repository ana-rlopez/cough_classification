from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import pandas as pd
import modelTrainingFunctions_original as modelTrainLib

def getClassification_scores(true_classes, predicted_classes):
    acc = accuracy_score(true_classes, predicted_classes)
    prec = precision_score(true_classes, predicted_classes,average="macro")
    f1 = f1_score(true_classes, predicted_classes,average="macro")
    recall = recall_score(true_classes, predicted_classes,average="macro")
    scores = [acc, prec, f1, recall]
    return scores


def evaluation_Step(pred_probs_perID):
    scores = getClassification_scores(pred_probs_perID['label'], pred_probs_perID['pred_class'])
    acc = scores[0]
    prec = scores[1]
    f1 = scores[2]
    recall = scores[3]
    
    return [acc, prec, f1, recall]
    
    #print('Scores:')
    #print('Accuracy: %f' % acc)
    #print('Precision: %f' % prec)
    #print('F1-score: %f' % f1)
    #print('Recall: %f' % recall)
    #      

    #confusion matrix
    conf_mat_df = pd.crosstab(mean_pred_probs['label'], mean_pred_probs['pred_class'], margins=True)
    
    print('\nConfusion matrix')
    print(conf_mat_df)
    return


def trainModels_getAvgEvalScores(nr_runs,X_train,y_train,ID_train,label_dict):

    all_acc =[]
    
    for i in range(nr_runs):
        
        pred_probs = modelTrainLib.modelTraining(X_train,y_train,ID_train)
        pred_probs_perID = modelTrainLib.get_predClass_per_audio(pred_probs, label_dict)
        
        scores = evaluation_Step(pred_probs_perID)
        
        acc = scores[0]
        print('Run %d: Accuracy=%f' % (i, acc))
        all_acc.append(acc)
               
       
    avg_acc = sum(all_acc)/len(all_acc)
    return avg_acc
        
        