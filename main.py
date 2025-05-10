from pipelines.ocr import run_ocr


if __name__ == "__main__":
    
    run_ocr(nb_shapes=4, selection=False)
    #run_ocr(nb_shapes=16, selection=False)
    #run_ocr(nb_shapes=16, selection=True)
    #run_ocr(nb_shapes=784, selection=False)
