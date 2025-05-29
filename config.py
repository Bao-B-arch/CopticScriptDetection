from pathlib import Path


# Définition de constantes

# Nombre de séparateurs pour l'affichage
NUMBER_SECTION_DEL = 50
# Chemin contenant les images coptes
DATABASE_PATH = Path("Images_Traitees")
# Chemin pour l'export des rapports et graphes
REPORT_PATH = Path("report")
EXPORT_PATH = Path("export")
GRAPH_PATH = Path("graph")
# Nom du fichier pour savegarder la base de données de features
SAVED_DATABASE_PATH = Path("database.npz")
# Random state pour les algorithmes faisant intervenir des probabilités
RANDOM_STATE = 0
# Train-test ratio pour l'entrainement des modèles
TEST_TRAIN_RATIO = 0.2
# NOMBRE_DE_PATCH = NB_LIGNE * NB_COLONNE = NB * NB
NB_SHAPE = 4
# lettres à retirer de l'analyse
LETTER_TO_REMOVE = ["Sampi", "Eta", "Psi", "Ksi", "Zeta"]
# Option pour changer le comportement du scripts
FORCE_COMPUTATION = True
FORCE_PLOT = False
FORCE_REPORT = True

IMAGE_SIZE = 28
BACKGROUND_COLOR = 77
FACTOR_SIZE_EXPORT = 100