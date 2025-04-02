import os
import subprocess
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from utils import Sets, outlier_analysis
from graph_utils import visualize_scaling, visualize_correlation, visualize_train_test_split, visualize_confusion_matrix

def generate_report(
        data_size: int,
        db_noscaling: Sets,
        db_scaled: Sets,
        train: Sets,
        test: Sets,
        models: dict,
        letter_to_remove: dict,
        random_state: int,
        show: bool = False,
        rep: bool = False
) -> None:

    report = {}
    report["DATABASE_DESCRIPTION"] = {
        "SIZE": data_size, 
        "LETTER_REMOVED": letter_to_remove,
        "TRAIN_SIZE": {"value": len(train.X), "percent": len(train.X) / data_size * 100},
        "TEST_SIZE": {"value": len(test.X), "percent": len(test.X) / data_size * 100},
        "TARGET": db_scaled.y.value_counts().to_dict(),
    }

    report["FEATURES_DESCRIPTION"] = {
        "NB_PATCHES": len(db_scaled.X.columns),
        "FEATURES": train.X.columns.to_list(),
        "CORRELATION_FEATURES": db_scaled.X.corr().to_dict(),
        "OUTLIERS_ANALYSIS": outlier_analysis(db_noscaling.X).to_dict(),
    }

    # Prédiction sur un échantillon de 5 données aléatoires
    sample_x = test.X.sample(n=10, random_state=random_state)
    sample_y = test.y.loc[sample_x.index]

    report["PREDICTION_EXEMPLE"] = {
        "REAL": sample_y.to_list()
    }
    for model_name, model in models.items():
        pred = model.predict(sample_x)
        report["PREDICTION_EXEMPLE"][model_name] = pred.tolist()

    # Vérifier à partir de score la qualité du modèle (avec l'ensemble de test)
    labels = np.unique(db_scaled.y)
    report["METRICS"] = {}
    cms = {}
    for model_name, model in models.items():
        pred_x = model.predict(test.X)
        ## accuracy
        acc = accuracy_score(test.y, pred_x)
        ## matthews corrcoeff
        mcc = matthews_corrcoef(test.y, pred_x)

        report["METRICS"][model_name] = {"ACCURACY": acc, "MCC": float(mcc)}

        ## confusion matrix
        cm = confusion_matrix(test.y, pred_x, labels=labels, normalize="true")
        cms[model_name] = cm

    # Génération des graphes
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    visualize_scaling(db_noscaling.X)
    visualize_correlation(db_scaled.X)
    visualize_train_test_split(train.y, test.y)
    for model_name, cm in cms.items():
        visualize_confusion_matrix(cm, labels, model_name)

    plt.tight_layout()
    if show: plt.show()
    
    if not os.path.exists("report"):
        os.makedirs("report")
    with open("report/report.yaml", "w", encoding="utf-8") as file:
        yaml.dump(report, file)

    test_quarto = subprocess.run(["quarto", "--version"], stdout = subprocess.DEVNULL)
    if rep & (test_quarto.returncode == 0):
        subprocess.run(["quarto", "render", ".\coptic_report.qmd", 
                        "--to", "pdf,revealjs", 
                        "--output-dir", "report\quarto"], 
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL)
