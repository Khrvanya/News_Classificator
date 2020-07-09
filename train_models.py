import joblib
import os
import copy
import re
import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from scrape_queries import process_text

CURR_PATH = os.path.abspath(os.curdir)
MODELS_PATH = os.path.join(CURR_PATH, 'models')


def get_vectorizers(feature=None) -> set:
    """
    Return list of text vectorizers
    """

    vectorizer = None

    if feature == 'all':
        vectorizer = {("cv", CountVectorizer(stop_words="english")),
                      ("tfidfv", TfidfVectorizer(stop_words="english"))}
    elif feature == 'cv':
        vectorizer = {("cv", CountVectorizer(stop_words="english"))}
    elif feature == 'tfidfv':
        vectorizer = {("tfidfv", TfidfVectorizer(stop_words="english"))}
    elif feature == None:
        vectorizer = set()

    assert not vectorizer == None, '!!!given feature is wrong!!!'

    return vectorizer


def get_classificators(feature=None) -> set:
    """
    Return list of classificators
    """

    classifier = None

    if feature == 'all':
        classifier = {("rf", RandomForestClassifier(n_jobs=-1, random_state=17)),
                      ("logreg", LogisticRegression(n_jobs=-1, random_state=17))}
    elif feature == 'rf':
        classifier = {("rf", RandomForestClassifier(n_jobs=-1, random_state=17))}
    elif feature == 'logreg':
        classifier = {("logreg", LogisticRegression(n_jobs=-1, random_state=17))}

    assert not classifier == None, '!!!given feature is wrong!!!'

    return classifier


def get_models_params(feature=None) -> dict:
    """
    Return dict of models params
    """

    param_dict = {}

    if feature == 'all' or feature == 'rf':
        rf_params = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [15, None],
            'rf__max_features': [.5, .7, 1]
        }
        param_dict['rf'] = rf_params

    if feature == 'all' or feature == 'logreg':
        logreg_params = {
            'logreg__C': np.logspace(-3, 4, 7),
        }
        param_dict['logreg'] = logreg_params

    if feature == 'all' or feature == 'cv':
        cv_params = {
            'cv__ngram_range': [(1, 2)],
            'cv__max_features': [1000, 1500, 2000, 3000]
        }
        param_dict['cv'] = cv_params

    if feature == 'all' or feature == 'tfidfv':
        tfidfv_params = {
            'tfidfv__ngram_range': [(1, 2)],
            'tfidfv__max_features': [1000, 1500, 2000, 3000]
        }
        param_dict['tfidfv'] = tfidfv_params

    assert (not param_dict == {}) or feature == None, '!!!given feature is wrong!!!'

    return param_dict


def find_best_pipeline(X, y, classificators: str, vectorizers=None, parameters=None):
    """
    Finds best model for data with vectorizer(not neccessary) and classifier
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=17)

    classifiers = get_classificators(classificators)
    vectorizers = get_vectorizers(vectorizers)
    params_dict = get_models_params(parameters)

    pipelines_scores = []

    for classifier in classifiers:
        vectorizers_copy = copy.deepcopy(vectorizers)

        while True:
            step = []

            if vectorizers_copy:
                step.append(vectorizers_copy.pop())
            step.append(classifier)
            pipeline = Pipeline(step)

            params = {}
            for element in step:
                try:
                    params.update(params_dict[element[0]])
                except KeyError:
                    pass

            grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            pipelines_scores.append((grid.best_score_, grid))

            if not vectorizers_copy:
                break

    best_grid = sorted(pipelines_scores, key=lambda x: -x[0])[0][1]
    best_pipe = best_grid.best_estimator_
    mask = best_grid.cv_results_['rank_test_score'] - 1
    best_cv_std = best_grid.cv_results_['std_test_score'][mask][0]

    #     assert best_cv_std < .1, '\n!!!std is bigger than 0.1!!!\n'                          #####

    print('!!!best pipeline steps: ', '-'.join(list(best_pipe.named_steps.keys())), '!!!')
    print('!!!best pipeline params: ', best_grid.best_params_, '!!!\n')
    print('!!!best mean and std cross-val score: ', best_grid.best_score_, ', ', best_cv_std, '!!!')
    print('!!!test score: ', best_pipe.score(X_test, y_test), '!!!\n')

    return best_pipe


def train_best_pipeline(queries_path: str, categories: list):
    """
    Makes and trains a model for a node
    Returns pipe of models that are trained
    """
    assert categories, '!!!categories list is empty!!!'
    
    full_data_text = load_files(queries_path, categories=categories,
                                encoding="utf-8", decode_error="replace", random_state=17)

    labels, counts = np.unique(full_data_text.target, return_counts=True)
    labels_sort = np.array([name.split(',')[0] for name in full_data_text.target_names])[labels]
    print('\n!!!making pipeline for node: \n', dict(zip(labels_sort, counts)), '!!!\n')

    X_text = [process_text(text) for text in full_data_text.data]
    y_text = full_data_text.target

    pipeline = find_best_pipeline(X_text, y_text, 'all', 'all', 'all')
    pipeline.fit(X_text, y_text)

    return pipeline


def make_classification_models(root_node, models_path, queries_path, delete_previous=False,
                                save=False):
    """
    Makes classification models for root structure using 
    scrape_articles.py
    """

    if not root_node.children_set:
        return None
    print('!!!start making', root_node.name, 'pipeline!!!')

    if delete_previous or not re.findall(root_node.name,
                                                   ' '.join(os.listdir(models_path))):

        trained_pipeline = train_best_pipeline(queries_path, root_node.get_children_queries())

        joblib.dump(trained_pipeline, os.path.join(models_path, root_node.name + '.sav'))

        if save:
            root_node.pipeline = trained_pipeline

    print('!!!end making', root_node.name, 'pipeline!!!\n\n')
    for node in root_node.children_set:
        make_classification_models(node, models_path, queries_path, delete_previous, save)


