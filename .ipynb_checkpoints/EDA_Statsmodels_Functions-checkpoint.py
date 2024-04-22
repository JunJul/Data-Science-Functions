# plot the confusion matrix
def confusionMatrixPlot(cm, labels):
    # cm = model.pred_table(threshold=0.5)
    cm_dis = ConfusionMatrixDisplay(cm, display_labels=labels)
    cm_dis.plot(cmap=plt.cm.Blues)

# The function takes a dataframe arugument to generate a matrix count plot @ Modified
def maxtrixCountPlot(input_data):
    rows = 4
    cols = 3
    row_count = col_count = 0
    fig, axe = plt.subplots(rows, cols, figsize=(20,15))
    
    for i in input_data.columns:
        sns.countplot(x = input_data[i],
                      ax=axe[row_count, col_count],
                      hue = input_data[i],
                      palette="flare")
        col_count += 1
    
        if col_count >= cols:
            col_count = 0
            row_count += 1
    
    fig.delaxes(axe[3, 1])
    fig.delaxes(axe[3, 2])


# statsmodels for logistic Regression
def statsLogModel(x, y):
    x = sm.add_constant(x)
    statsLogitModel = sm.Logit(y, x).fit()
    
    return statsLogitModel

