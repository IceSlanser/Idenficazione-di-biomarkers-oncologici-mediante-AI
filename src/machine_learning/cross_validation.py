from typing import Callable
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_val_score

# Create objective for Optuna study
def create_objective(X_train:pd.DataFrame, y_train:pd.Series) -> Callable[[optuna.Trial], float]:
    print("Creating objective function...")
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 50),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            'seed': 42
        }

        # Cross validation with defined params
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=2)  # TODO: cv=5

        # Return CV's mean scores
        return scores.mean()

    return objective

# Combine Bayesian optimization with Cross Validation
def params_optimizedBCV(X_train:pd.DataFrame, y_train:pd.Series):
    # Create the callable objective
    objective = create_objective(X_train, y_train)



    # Optimization with 50 trials using Optuna
    print("Staring Optuna study...")
    study = optuna.create_study(direction='maximize', storage="sqlite:///optimized_study.db",
                                study_name="my_study", load_if_exists=True)
    study.optimize(objective, n_trials=50, n_jobs=2)  # TODO: n_trials=50

    return study