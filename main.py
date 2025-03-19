import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import data_loading

# Scikit learn model
from sklearn.ensemble import RandomForestClassifier

# Scikit learn validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split 

# Constante
NUMBER_SECTION_DEL = 50
DATABASE_PATH = "..//..//Images_Traitees"

if __name__ == "__main__":
    raw_data = data_loading.load_database(DATABASE_PATH)
    means_data = data_loading.mean_grayscale(raw_data)

    # Clean data
    # Dropna drops missing values (think of na as "not available")
    means_data = means_data.dropna(axis=0)

    # Target
    y = means_data.loc[:, "Letter"]

    # Features
    means_features = ["Mean"]
    X = means_data.loc[:, means_features]
    print(
f"FEATURE NAME:\n\
{means_features}\n\
{'-'*NUMBER_SECTION_DEL}"
)

    # Split between Train and Test
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    print(
f"DATA SIZE:\n\
TOTAL DATA SIZE:\t{len(X)}\n\
TRAIN SIZE:\t\t{len(train_X)} | POURCENTAGE:{(len(train_X) / len(X) * 100):.2f}%\n\
TEST SIZE:\t\t{len(test_X)} | POURCENTAGE:{(len(test_X) / len(X) * 100):.2f}%\n\
{'-'*NUMBER_SECTION_DEL}"
    )

    df_set = pd.concat([
        pd.DataFrame({"Letter": train_y, "Set": "Train"}), 
        pd.DataFrame({"Letter": test_y, "Set": "Test"})
    ])
    df_set.value_counts().plot(kind="bar", stacked=True)
    plt.show()

    # models
    rfc = RandomForestClassifier()
  
    # Entrainement
    # il y a un algo d'optimisation sur un critère 
    rfc.fit(train_X, train_y)
        
    # prediction
    sample_data = means_data.sample(n=5, random_state=0)
    X_sample = sample_data.loc[:, means_features]
    y_sample = sample_data.loc[:, "Letter"]
    prediction_rfc = rfc.predict(X_sample)
       
    # Visualization
    pretty_format = lambda table: np.array2string(table, formatter={'float_kind': lambda x: f'{x:.1f}'})
    print("EXAMPLE:")
    print(f"REAL:\t{pretty_format(y_sample.to_numpy())}")
    print(f"RFC:\t{pretty_format(prediction_rfc)}")
    print("-"*NUMBER_SECTION_DEL)
    
    # Model validation
    acc_rfc = accuracy_score(
        test_y,
        rfc.predict(test_X)
    )

    # Affichage des résultats 
    print("METRICS:")
    print(f"ACC RFC:\t{acc_rfc:.2f}")
    print('-'*NUMBER_SECTION_DEL)
     
    # TODO 
    ## utiliser SVM classifier 
    ## créer des meilleurs features (hog)
    ## meilleur analyse des metrics
    ## bootstrap
