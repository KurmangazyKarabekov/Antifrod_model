def lift_metric_plot(model, X_test, y_test):
    all_pred = pd.DataFrame()
    all_pred["prob"] = model.predict_proba(X_test)[:, 1]
    all_pred["true"] = y_test.values
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    y_true = all_pred["true"]
    recall = []
    precision = []
    score = []
    f_score = []

    aa = -np.sort(
        -all_pred["prob"].describe(percentiles=np.linspace(0, 1, 101))[4:105].values
    )
    for i in range(0, len(aa)):
        f = lambda x: 1 if x >= aa[i] else 0
        score.append(aa[i])
        y_pred = all_pred["prob"].map(f)
        g = recall_score(y_true, y_pred, average="binary")
        recall.append(g)
        p = precision_score(y_true, y_pred, average="binary")
        precision.append(p)
        t = f1_score(y_true, y_pred, average="binary")
        f_score.append(t)

    temp = pd.DataFrame()
    temp["score"] = score
    temp["recall"] = recall
    temp["precision"] = precision
    temp["top_percentile"] = np.round(list(np.linspace(0, 100, 101))).tolist()

    temp["lift"] = temp["recall"] / (temp["top_percentile"] / 100.0)
    temp["max_lift"] = 1 / (y_test.sum() / len(y_test))
    temp["relative_lift"] = temp["lift"] / temp["max_lift"]

    temp["percent"] = 100 * (y_test.sum() / len(y_test))
    plt.figure()
    plt.plot(temp[1:]["top_percentile"], temp[1:]["lift"])
    plt.xlabel("Percentile")
    plt.ylabel("LIFT")
    plt.scatter(
        1,
        temp[temp["top_percentile"] == 1]["lift"][1],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        1,
        temp[temp["top_percentile"] == 1]["lift"][1],
        "      top 1 : {}".format(
            np.round(temp[temp["top_percentile"] == 1]["lift"][1], 3)
        ),
        fontsize=10,
    )
    plt.scatter(
        10,
        temp[temp["top_percentile"] == 10]["lift"][10],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        10,
        temp[temp["top_percentile"] == 10]["lift"][10],
        "      top 10 : {}".format(
            np.round(temp[temp["top_percentile"] == 10]["lift"][10], 3)
        ),
        fontsize=10,
    )
    plt.title("Lift-chart")
    plt.figure()
    plt.plot(temp[1:]["top_percentile"], temp[1:]["relative_lift"])
    plt.xlabel("Percentile")
    plt.ylabel("Relative_lift")
    plt.scatter(
        1,
        temp[temp["top_percentile"] == 1]["relative_lift"][1],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        1,
        temp[temp["top_percentile"] == 1]["relative_lift"][1],
        "      top 1 : {}".format(
            np.round(temp[temp["top_percentile"] == 1]["relative_lift"][1], 3)
        ),
        fontsize=10,
    )
    plt.scatter(
        10,
        temp[temp["top_percentile"] == 10]["relative_lift"][10],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        10,
        temp[temp["top_percentile"] == 10]["relative_lift"][10],
        "      top 10 : {}".format(
            np.round(temp[temp["top_percentile"] == 10]["relative_lift"][10], 3)
        ),
        fontsize=10,
    )
    plt.title("Relative-lift-chart")
    plt.figure()
    plt.plot(temp[1:]["top_percentile"], temp[1:]["recall"])
    plt.xlabel("Percentile")
    plt.ylabel("GAIN")
    plt.scatter(
        10,
        temp[temp["top_percentile"] == 10]["recall"][10],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        10,
        temp[temp["top_percentile"] == 10]["recall"][10],
        "      top 10 : {}".format(
            np.round(temp[temp["top_percentile"] == 10]["recall"][10], 3)
        ),
        fontsize=10,
    )
    plt.scatter(
        40,
        temp[temp["top_percentile"] == 40]["recall"][40],
        marker="o",
        s=30,
        zorder=10,
        color="b",
    )
    plt.text(
        40,
        temp[temp["top_percentile"] == 40]["recall"][40],
        "      top 40 : {}".format(
            np.round(temp[temp["top_percentile"] == 40]["recall"][40], 3)
        ),
        fontsize=10,
    )
    plt.title("Gain-chart")
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_test.values, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=1, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
    print(temp[1:].head(10))
    return temp[1:]
