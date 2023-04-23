Phteven
==============================

Developing a model to detect rotting meat in supermarkets

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── flask_demo         <- A web demo for this project created using flask
    │   ├── models         <- Contains resnet 50-fe model binary
    │   ├── screenshots    <- Screenshots of demo web pages
    │   ├── static         <- css and static resources
    │   └── Procfile       <- Process file to run flask server
    │   └── README.md      <- Description of flask demo
    │   └── app.yaml       <- Config file for environment setup
    │   └── main.py        <- Main flask file for demo
    │   └── requirements.txt     <- Package installation
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks used for modelling, predictions and experiments
    │   └── 1.0-EDA-and-preprocessing.ipynb
    │   └── 2.0-image-segmentation.ipynb
    │   └── 3.0-resnet_model_evaluation.ipynb
    │   └── 4.0-resnet-LIME-Explanation.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   ├── evaluation           <- Scripts for model evaluation
    │   │   └── misclassification_cost.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_resnet.py
    │   │   └── predict_resnet.py
    │   │   └── resnet_pipe.sh
    │   │   └── resnet_predict.sh 
    │   │
    │   └── utils  <- Scripts for utilitity functions
    │   │   └── manage_constants.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
