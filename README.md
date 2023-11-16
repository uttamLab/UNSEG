# UNSEG for Unsupervised Segmentation of Cells and Their Nuclei in Tissue Images

This is an implementation of [UNSEG](https://www.biorxiv.org/) on Python 3 with using  scikit-image, OpenCV, scikit-learn, and SciPy. The algorithm generates two mutually-consistent segmentation masks for cells and their nuclei in images of complex tissue samples. 

![Segmentation Example](content/unseg_segmentation.png)

The repository includes:
* Source Python code of UNSEG.
* Jupyter notebook to demonstrate the nuclei and cell segmention pipeline.
* Zipped test RGB image of human gallbladder tissue, where the blue and red channels contain nucleus and cell membrane markers, respectively.
* Requirements.txt

* If you find this code useful in your research, please consider citing:
  @inproceedings{plummerCITE2018,
Author = {Bryan A. Plummer and Paige Kordas and M. Hadi Kiapour and Shuai Zheng and Robinson Piramuthu and Svetlana Lazebnik},
Title = {Conditional Image-Text Embedding Networks},
Booktitle  = {The European Conference on Computer Vision (ECCV)},
Year = {2018}
}
