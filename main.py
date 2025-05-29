import argparse
from timeit import default_timer as timer

from pipelines.ocr import run_ocr
from pipelines.selection import run_selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--selection", action="store_true")
    parser.add_argument("-r", "--run", action="store_true")
    args = parser.parse_args()

    main_start_timer = timer()

    if args.selection:
        # utilisé pour lancer la pipeline analysant les sélections de features
        run_selection(nb_shapes=16, selection=["anova", "mi", "l1", "rfe"])
    if args.run:
        # utilisé pour lancer la pipeline d'entrainement du modèle
        run_ocr(nb_shapes=16, selection=None)
        run_ocr(nb_shapes=16, selection="l1")
        run_ocr(nb_shapes=784, selection=None)

    main_end_timer = timer()
    print(f"Script lasted {main_end_timer - main_start_timer:.2f} seconds.")
