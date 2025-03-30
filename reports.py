import yaml
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from utils import Sets, outlier_analysis
from graph_utils import visualize_scaling, visualize_correlation, visualize_train_test_split, visualize_correlation_matrix

def generate_report(
        data_size: int,
        db_noscaling: Sets,
        db_scaled: Sets,
        train: Sets,
        test: Sets,
        models: dict,
        random_state: int
) -> None:

    report = {}
    report["DATABASE_DESCRIPTION"] = {
        "SIZE": data_size, 
        "TRAIN_SIZE": {"value": len(train.X), "percent": len(train.X) / data_size * 100},
        "TEST_SIZE": {"value": len(test.X), "percent": len(test.X) / data_size * 100},
        "NB_PATCHES": len(db_scaled.X.columns)
    }

    report["MODEL_DESCRIPTION"] = {
        "FEATURES": train.X.columns.to_list(),
        "TARGET": db_scaled.y.value_counts().to_dict(),
        "CORRELATION_FEATURES": db_scaled.X.corr().to_dict(),
        "OUTLIERS ANALYSIS": outlier_analysis(db_noscaling.X).to_dict(),
    }

    visualize_scaling(db_noscaling.X)
    visualize_correlation(db_scaled.X)
    visualize_train_test_split(train.y, test.y)

    # Prédiction sur un échantillon de 5 données aléatoires
    sample_x = test.X.sample(n=10, random_state=random_state)
    sample_y = test.y.loc[sample_x.index]

    report["PREDICTION EXEMPLE"] = {
        "REAL": sample_y.to_list()
    }
    for model_name, model in models.items():
        pred = model.predict(sample_x)
        report["PREDICTION EXEMPLE"][model_name] = pred.tolist()

    # Vérifier à partir de score la qualité du modèle (avec l'ensemble de test)
    labels = np.unique(db_scaled.y)
    report["METRICS"] = {}
    for model_name, model in models.items():
        pred_x = model.predict(test.X)
        ## accuracy
        acc = accuracy_score(test.y, pred_x)

        ## matthews corrcoeff
        mcc = matthews_corrcoef(test.y, pred_x)

        ## confusion matrix
        cm = confusion_matrix(test.y, pred_x, labels=labels, normalize="true")
        visualize_correlation_matrix(cm, labels, model_name)

        report["METRICS"][model_name] = {"ACCURACY": acc, "MCC": float(mcc)}

    with open("report.yaml", "w", encoding="utf-8") as file:
        yaml.dump(report, file)
