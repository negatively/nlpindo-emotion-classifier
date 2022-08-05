# train.py
import config
import os

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE)
    
    mapping = {
        "neutral":0,
        "joy":1,
        "sadness":2,
        "fear":3,
        "surprise":4,
        "anger":5,
        "shame":6,
        "disgust":7
    }

    df.loc[:, "Emotion"] = df["Emotion"].map(mapping)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['Emotion'].values
    kf = StratifiedKFold(n_splits=5)
    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_,'kfold'] = f
    
    for fold_ in range(5):
        # Split Data
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        X_train = train_df["Text"]
        y_train = train_df["Emotion"]
        X_test = test_df["Text"]
        y_test = test_df["Emotion"]

        # Pipeline
        pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
        pipe_lr.fit(X_train, y_train)

        preds = pipe_lr.score(X_test, y_test)

        print(f"Fold: {fold_}")
        print(f"Accuracy : {preds}")

        # save the model
        joblib.dump(pipe_lr, os.path.join(config.MODEL_OUTPUT, f"model_{fold_}.pkl"))




