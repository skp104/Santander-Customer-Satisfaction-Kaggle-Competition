import pandas as pd
from sklearn import cross_validation as cv
import xgboost as xgb

def XGB():
    df_train = pd.read_csv("train.csv")
    df_test  = pd.read_csv("test.csv")   

    df_train = df_train.replace(-999999,2)

    id_test = df_test['ID']
    y_train = df_train['TARGET'].values
    X_train = df_train.drop(['ID','TARGET'], axis=1).values
    X_test = df_test.drop(['ID'], axis=1).values

    clf = xgb.XGBClassifier(objective='binary:logistic',
                    missing=9999999999,
                    max_depth = 7,
                    n_estimators=200,
                    learning_rate=0.1, 
                    nthread=4,
                    subsample=1.0,
                    colsample_bytree=0.5,
                    min_child_weight = 3,
                    reg_alpha=0.01,
                    seed=7)

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    scores = cv.cross_val_score(clf, X_train, y_train) 

    submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
    return submission
    #submission.to_csv("submission_XGB.csv", index=False)