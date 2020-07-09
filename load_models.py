import numpy as np
import joblib
import os

MIN_POSSIBILITY = .35
CURR_PATH = os.path.abspath(os.curdir)
MODELS_PATH = os.path.join(CURR_PATH, 'models')

from node_structure import make_node_structure, Node
ROOT_NODE = make_node_structure(MODELS_PATH)


def article_classify(article: str, node: Node, diff_coef=.1) -> set:
    """
    Gets an article text and Node with pipeline
    Returns the set of string names of categories which the article belongs to
    
    params article: the article that has to be classified
    params node: has downloaded pipeline
    params diff_coef: the difference between the max category result and the others
    """
    
    try:
        y_pred = node.pipeline.predict_proba(article).reshape(-1)
                
        mask = y_pred > (y_pred.max() - diff_coef)
        classes = sorted(list(node.get_children_names())) 
        # other PCs may had loaded sets differently, not in alphabet order, find out how
                
        result = set(np.array(classes)[mask]) if y_pred.max() >= MIN_POSSIBILITY else set()
        
    except Exception as error:
        print('node name: ', node.name) 
        print('error: ', error)
        assert node.pipeline == 'Error! No path exists!', '!!!No such pipeline loaded!!!'
        result = {}
    
    return result
    
    
def article_tree_classify(article: list, node=ROOT_NODE, diff_coef=.1) -> dict:
    """
    Classifies an article above all the node structure
    Gets Node structure, article, diff_coef, which is metioned above
    Returns a dictionary, which is used rucursively to create categories and 
    subcategories and ... which this article is in
    example: {'energy': {'crude oil': {}}}
    
    params article:  the text article that has to be classified
    params node: a tree structure that classifies the article
    params diff_coef: the difference between the max category result and the others
    """
    
    if not node.children_set:
        return {}
    
    diction = {}
    result_categories = article_classify(article, node, diff_coef)
    
    while result_categories:
        category = result_categories.pop()
        
        for node in node.children_set:
            
            if category == node.name:
                diction[category] = article_tree_classify(article, node)            
 
    return diction
    
    
text_path = r'queries\energy, energy, energy finance, energy news, energy stocks, energy futures, energy shares\2 Top Energy Stocks to Buy Now.txt'
text = [open(text_path, 'r').read()    ## text = [str]
print(article_tree_classify(text))