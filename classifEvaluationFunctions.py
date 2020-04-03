from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import pandas as pd

def getClassification_scores(true_classes, predicted_classes):
    acc = accuracy_score(true_classes, predicted_classes)
    prec = precision_score(true_classes, predicted_classes,average="macro")
    f1 = f1_score(true_classes, predicted_classes,average="macro")
    recall = recall_score(true_classes, predicted_classes,average="macro")
    scores = [acc, prec, f1, recall]
    return scores


def evaluation_Step(mean_pred_probs):
    scores = getClassification_scores(mean_pred_probs['label'], mean_pred_probs['pred_class'])
    acc = scores[0]
    prec = scores[1]
    f1 = scores[2]
    recall = scores[3]
    
    print('Scores:')
    #print('Accuracy: %f' % acc)
    print('Precision: %f' % prec)
    print('F1-score: %f' % f1)
    print('Recall: %f' % recall)
          

    #confusion matrix
    conf_mat_df = pd.crosstab(mean_pred_probs['label'], mean_pred_probs['pred_class'], margins=True)
    
    print('\nConfusion matrix')
    print(conf_mat_df)
    return


