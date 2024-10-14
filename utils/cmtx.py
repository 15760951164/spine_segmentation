
from sklearn.metrics import confusion_matrix,roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import pandas as pd

def get_confusion_matrix(preds, labels, num_classes, normalize="true"):

    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    # if labels.ndim == preds.ndim:
    #     labels = torch.argmax(labels, dim=-1)
    # # Get the predicted class indices for examples.
    # preds = torch.flatten(torch.argmax(preds, dim=-1))
    # labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    
    #print(cmtx)
    
    return cmtx

def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure

def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
):
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
        )
        # Add the confusion matrix image to writer.
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)
        
def add_roc_curve(score_list , label_list, writer, tag, step, num_classes, class_names, output_path=None):

    # if isinstance(preds_socer, list):
    #     preds_socer = torch.cat(preds_socer, dim=0)
    # if isinstance(labels, list):
    #     labels = torch.cat(labels, dim=0)
    
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_classes)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Class_{}_AUC={:.2f}'.format(class_names[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='Average ROC curve (Area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)
    
    
    if output_path is not None:
        data = pd.DataFrame({'fpr_avg': fpr["micro"], 
                             'tpr_avg': tpr["micro"],
                             'auc' : np.array(roc_auc["micro"])
                             })
        data.to_excel(output_path, index=False)
    
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    writer.add_figure(tag=tag, figure=plt.gcf(), global_step=step)
    plt.close()