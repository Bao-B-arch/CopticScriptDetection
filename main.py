from timeit import default_timer as timer

from pipelines.ocr import run_ocr
from pipelines.selection import run_selection


if __name__ == "__main__":
    main_start_timer = timer()
    run_selection(nb_shapes=16, selection=True)
    # run_ocr(nb_shapes=4, selection=False)
    # run_ocr(nb_shapes=16, selection=False)
    # run_ocr(nb_shapes=16, selection=True)
    # run_ocr(nb_shapes=784, selection=False)

    main_end_timer = timer()
    print(f"Script lasted {main_end_timer - main_start_timer:.2f} seconds.")
