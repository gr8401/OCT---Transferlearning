from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# List with true labels, 250 from each class (CNV, DME, DRUSEN, NORMAL)
y_true = [];
for i in range(0,1000):
    if (i < 250):
        y_true.append("CNV")   
    elif (i > 249 and i < 500):
        y_true.append("DME")
    elif (i > 499 and i < 750):
        y_true.append("DRUSEN")
    elif (i > 749 and i < 1000):
        y_true.append("NORMAL")


'''
# OBS denne y_true skal udkommenteres når vi når til hele datasættet skal testes
y_true = ["CNV", "CNV", "CNV", "CNV", "CNV", "CNV", "CNV", "CNV", "CNV", "CNV", 
          "DME", "DME", "DME", "DME", "DME", "DME", "DME", "DME", "DME", "DME", 
          "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", "DRUSEN", 
          "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"]
'''


#OBS tjek det er den rigtige matrices med alle prædikterede værdier i vi bruger til endelig test
y_pred = class_by_name[:]

''' Calculate Confusion Matrix and define possible classes '''
cm = confusion_matrix(y_true, y_pred, labels=["CNV", "DME", "DRUSEN", "NORMAL"])
classlabels = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Calculate to sum of predicted for each row/class and find the value in percent
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)
            
''' Overall evaluation - all four classes together '''
accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
misclass = 1 - accuracy
precision = precision_score(y_true, y_pred, labels=classlabels, average='weighted', sample_weight=None)
recall = recall_score(y_true, y_pred, labels=classlabels, average='weighted', sample_weight=None)

''' Evaluation for each class (ec) '''
print(classification_report(y_true, y_pred, target_names=classlabels))

# TP, TN, FP, FN
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)
tp_float = tp.astype(float)
fp_float = fp.astype(float)
fn_float = fn.astype(float)
tn_float = tn.astype(float)

# Sensitivitet (recall, TPR), specificity (TNR), precison (PPV), NPV, accuracy
recall_ec = tp/(tp+fn)
speci_ec = tn/(tn+fp)
precision_ec = tp/(tp+fp)
NPV_ec = tn/(tn+fn)
accuracy_ec = (tp+tn)/(tp+fp+fn+tn)

#Øvrige mål: fall out (FPR), FNR, FDR
FPR_ec = fp/(fp+tn)
FNR_ec = fn/(tp+fn)
FDR_ec = fp/(tp+fp)

''' Plot incl. labels, title and ticks '''
plt.figure(0)
ax= plt.subplot()
sns.heatmap(cm, cmap='Blues', annot=annot, fmt='', ax=ax, cbar=False)
ax.set_xlabel('Predicted labels\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy,misclass));
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(classlabels); ax.yaxis.set_ticklabels(classlabels);

''' ROC curve '''
# Binarize the true classes
y = label_binarize(y_true, classes=["CNV", "DME", "DRUSEN", "NORMAL"])
#classes_pred_binary = label_binarize(Classes_pred, classes=["CNV", "DME", "DRUSEN", "NORMAL"])
n_classes = y.shape[1]

# CNV
y_true_cnv = y[:,0]
preds_cnv = Pred_prob_cat[:,0] #præcise sandsynlighedsværdier fx 0,99, åben over for thresholds
#preds_binary_cnv = classes_pred_binary[:,0] #0 eller 1, fast threshold
fpr_cnv, tpr_cnv, threshold_cnv = roc_curve(y_true_cnv, preds_cnv, drop_intermediate=False)
#fpr_cnv, tpr_cnv, threshold_cnv = roc_curve(y_true_cnv, preds_binary_cnv)
roc_auc_cnv = roc_auc_score(y_true_cnv, preds_cnv)
#roc_auc_cnv = auc(fpr_cnv, tpr_cnv) #tidligere AUC beregning

plt.figure(1)
plt.title('ROC Curve for Classification of CNV Images')
plt.plot(fpr_cnv, tpr_cnv, 'b', label = 'AUC = %0.4f' % roc_auc_cnv)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# DME
y_true_dme = y[:,1]
preds_dme = Pred_prob_cat[:,1]
#preds_binary_dme = classes_pred_binary[:,1]
fpr_dme, tpr_dme, threshold_dme = roc_curve(y_true_dme, preds_dme, drop_intermediate=False)
#fpr_dme, tpr_dme, threshold_dme = roc_curve(y_true_dme, preds_binary_dme)
roc_auc_dme = roc_auc_score(y_true_dme, preds_dme)

plt.figure(2)
plt.title('ROC Curve for Classification of DME Images')
plt.plot(fpr_dme, tpr_dme, 'b', label = 'AUC = %0.4f' % roc_auc_dme)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# DRUSEN
y_true_drusen = y[:,2]
preds_drusen = Pred_prob_cat[:,2]
#preds_binary_drusen = classes_pred_binary[:,2]
fpr_drusen, tpr_drusen, threshold_drusen = roc_curve(y_true_drusen, preds_drusen, drop_intermediate=False)
#fpr_drusen, tpr_drusen, threshold_drusen = roc_curve(y_true_drusen, preds_binary_drusen)
roc_auc_drusen = roc_auc_score(y_true_drusen, preds_drusen)


plt.figure(3)
plt.title('ROC Curve for Classification of DRUSEN Images')
plt.plot(fpr_drusen, tpr_drusen, 'b', label = 'AUC = %0.4f' % roc_auc_drusen)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# NORMAL
y_true_normal = y[:,3]
preds_normal = Pred_prob_cat[:,3]
#preds_binary_normal = classes_pred_binary[:,3]
fpr_normal, tpr_normal, threshold_normal = roc_curve(y_true_normal, preds_normal, drop_intermediate=False)
#fpr_normal, tpr_normal, threshold_normal = roc_curve(y_true_normal, preds_binary_normal)
roc_auc_normal = roc_auc_score(y_true_normal, preds_normal)

plt.figure(4)
plt.title('ROC Curve for Classification of NORMAL Images')
plt.plot(fpr_normal, tpr_normal, 'b', label = 'AUC = %0.4f' % roc_auc_normal)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





