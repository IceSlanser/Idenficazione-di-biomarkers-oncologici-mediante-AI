import optuna
import pandas as pd
from fontTools.misc.textTools import tostr
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from src.machine_learning.cross_validation import params_optimizedBCV


def split_data(dataframe: pd.DataFrame, targetColumn:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("Splitting data...")
    X =dataframe.drop(columns=targetColumn, axis=1).copy()
    y = dataframe[targetColumn].copy()

    ## Verify y has only 1 or 0 as value
    # print(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ## Verify stratify
    # print(sum(y) / len(y))
    # print(sum(y_train) / len(y_train))
    # print(sum(y_test) / len(y_test))

    return X_train, X_test, y_train, y_test

def build_XGB_model(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series) -> None:
    print("Building XGBoost model...")

    ## New study
    # best_params = params_optimizedBCV(X_train, y_train).best_params

    study = optuna.load_study(study_name="my_study", storage="sqlite:///optimized_study.db")
    best_params = study.best_params
    print(best_params)
    clf_xgb = xgb.XGBClassifier(**best_params, n_jobs=2)
    clf_xgb.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=True)

    ## Display confusion matrix
    print("Building confusion matrix...")
    predictions = clf_xgb.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Ill"])
    disp.plot()
    plt.show() # show the confusion matrix

    # bst = clf_xgb.get_booster()
    # for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    #     print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

    print("Building graph...")
    node_params = {'shape': 'box',  # Make the nodes fancy
                   'style': 'filled, rounded',
                   'fillcolor': '#78cbe'}
    leaf_params = {'shape': 'box',
                   'style': 'filled, rounded',
                   'fillcolor': '#e48038'}

    ## Show the tree
    # NOTE: num_trees does not indicate the number of trees, but the specific tree
    graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
                                 condition_node_params=node_params,
                                 leaf_node_params=leaf_params,)
    graph_data.view(filename='XGBTree_breast_cancer')  # Save as pdf
    # print(df.head())