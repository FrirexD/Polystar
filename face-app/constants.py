###################################################################################################
#                                                                                                 #
#       Every relative path must follow this directory configuration in order to work properly    #     
#                                                                                                 #
###################################################################################################

DATA_DIR = "./data/"
OUTPUT_DIR = "./data/output/"
CELEBA_DIR = "./data/celebA/"
SAMPLES_DIR = "./data/samples/"
PREPOC_DIR = "./data/preprocessed/"
DEFAULT_BATCH_SIZE = 100

"""
# Current container directory configuration

.
└── app/
    ├── app.py
    ├── requierements.txt
    ├── data/
    │   ├── output/
    │   │   ├── test1.jpg
    │   │   └── ...
    │   ├── celebA/
    │   |   ├── 000001.jpg
    │   |   ├── ...
    |   |   └── 000999.jpg 
    │   └── samples/
    │       ├── image_sample1.jpg
    │       └── ...
    └── models/
        └── buffalo_l.zip

"""