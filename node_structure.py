from scrape_dataset import make_filename_safe
import os
import joblib

BASIC_LINKS_AMOUNT = 300

def to_list(obj) -> list:
    if type(obj) == list:
        return obj
    return [obj]
    

class Node:
    def __init__(self, name: str, queries=None, links_number=None, children_set=None):
        if children_set is None:
            children_set = set()
        if queries is None:
            queries = [name]
        if links_number is None:
            links_number = [BASIC_LINKS_AMOUNT] * len(queries)

        self.name = name
        self.queries = to_list(queries)
        self.links_number = to_list(links_number)
        self.children_set = children_set  # set of Nodes or None
        self.pipeline = None
        
    def load_pipeline(self, pipeline_folder_path: str):
        path = os.path.join(pipeline_folder_path, self.name + '.sav')
        if os.path.exists(path):
            self.pipeline = joblib.load(path)
        else:
            self.pipeline = 'Error! No path exists!'

    def get_children_names(self):
        if self.children_set == None:
            return None
        name_set = set()
        for node in self.children_set:
            name_set.add(node.name)
        return name_set

    def __str__(self):
        print('name: ', self.name, '\nqueries: ', self.queries,
              '\nlinks number: ', self.links_number,
              '\nchildren names: ', self.get_children_names(),
              '\npipeline: ', self.pipeline)
        return '\n'


def load_node_structure_models(root_node: Node, pipeline_folder_path: str) -> Node:
    """
    Loads pipelines to node and all its children and ...
    """
    
    if not root_node.children_set:
        return None
    
    root_node.load_pipeline(pipeline_folder_path)
    
    for node in root_node.children_set:
        node.load_pipeline(pipeline_folder_path)
        load_node_structure_models(node, pipeline_folder_path)


def make_node_structure(pipeline_folder_path=None) -> Node:
    """
    Makes a Node structure, which looks like a tree
    Returns the tree root
    """

    corn = Node('corn', ['corn', 'corn agriculture', '"corn"', 'corn news', 'corn stocks', 'corn futures', 'corn shares'], 0)
    soybean = Node('soybean', ['soybean', 'soybean agriculture', '"soybean"', 'soybean news', 'soybean stocks', 'soybean futures', 'soybean shares'], 0)
    cattle = Node('cattle', ['cattle', 'cattle agriculture', '"cattle"', 'cattle news', 'cattle stocks', 'cattle futures', 'cattle shares'], 0)
    sugar = Node('sugar', ['sugar', 'sugar agriculture', '"sugar"', 'sugar news', 'sugar stocks', 'sugar futures', 'sugar shares'], 0)
    agriculture = Node('agriculture', ['agriculture', '"agriculture"', 'agriculture finance', 'agriculture news', 'agriculture stocks', 'agriculture futures', 'agriculture shares'],
                       0, {corn, soybean, cattle, sugar})

    bitcoin = Node('bitcoin', ['bitcoin', 'bitcoin crypto', '"bitcoin"', 'bitcoin news', 'bitcoin stocks', 'bitcoin futures', 'bitcoin shares'], 0)
    dash = Node('dash', ['dash', 'dash crypto', '"dash"', 'dash news', 'dash stocks', 'dash futures', 'dash shares'], 0)
    ethereum = Node('ethereum', ['ethereum', 'ethereum crypto', '"ethereum"', 'ethereum news', 'ethereum stocks', 'ethereum futures', 'ethereum shares'], 0)
    litecoin = Node('litecoin', ['litecoin', 'litecoin crypto', '"litecoin"', 'litecoin news', 'litecoin stocks', 'litecoin futures', 'litecoin shares'], 0)
    monero = Node('monero', ['monero', 'monero crypto', '"monero"', 'monero news', 'monero stocks', 'monero futures', 'monero shares'], 0)
    ripple = Node('ripple', ['ripple', 'ripple crypto', '"ripple"', 'ripple news', 'ripple stocks', 'ripple futures', 'ripple shares'], 0)
    cryptocurrency = Node('cryptocurrency',['cryptocurrency',  'crypto', '"cryptocurrency"', 'cryptocurrency finance', \
                    'cryptocurrency news', 'cryptocurrency stocks', 'cryptocurrency futures', 'cryptocurrency shares'],
                          0, {bitcoin, dash, ethereum, litecoin, monero, ripple})

    gold = Node('gold', ['gold', 'gold metals', '"gold"', 'gold news', 'gold stocks', 'gold futures', 'gold shares'], 0)
    silver = Node('silver', ['silver', 'silver metals', '"silver"', 'silver news', 'silver stocks', 'silver futures', 'silver shares'], 0)
    platinum = Node('platinum', ['platinum', 'platinum metals', '"platinum"', 'platinum news', 'platinum stocks', 'platinum futures', 'platinum shares'], 0)
    copper = Node('copper', ['copper', 'copper metals', '"copper"', 'copper news', 'copper stocks', 'copper futures', 'copper shares'], 0)
    metals = Node('metals', ['metals', 'metals finance', '"metals"', 'metals news', 'metals stocks', 'metals futures', 'metals shares'],
                  0, {gold, silver, platinum, copper})

    crude_oil = Node('crude oil', ['crude oil', 'crude oil energy', '"crude oil"', 'crude oil news', 'crude oil stocks', 'crude oil futures', 'crude oil shares'], 0)
    natural_gas = Node('natural gas', ['natural gas', 'natural gas energy', '"natural gas"', 'natural gas news', 'natural gas stocks', 'natural gas futures', 'natural gas shares'], 0)
    brent_crude = Node('brent crude', ['brent crude', 'brent crude energy', '"brent crude"', 'brent crude news', 'brent crude stocks', 'brent crude futures', 'brent crude shares'], 0)
    energy = Node('energy', ['energy', '"energy"', 'energy finance', 'energy news', 'energy stocks', 'energy futures', 'energy shares'],
                  0, {crude_oil, natural_gas, brent_crude})

    nyse_composite = Node('nyse composite',
                          ['NYSE Composite', 'NYSE Composite index', '"NYSE Composite"', 'NYSE Composite news', 'NYSE Composite stocks', 'NYSE Composite futures', 'NYSE Composite shares'], 0)
    s_p500 = Node('s&p500', ['S%26P 500', 'S%26P 500 index', '"S%26P 500"', 'S%26P 500 news', 'S%26P 500 stocks', 'S%26P 500 futures', 'S%26P 500 shares'], 0)
    nasdaq100 = Node('nasdaq100', ['Nasdaq-100', 'Nasdaq-100 index', '"Nasdaq-100"', 'Nasdaq-100 news', 'Nasdaq-100 stocks', 'Nasdaq-100 futures', 'Nasdaq-100 shares'], 0)
    russell2000 = Node('russell2000', ['Russell 2000', 'Russell 2000 index', '"Russell 2000"', 'Russell 2000 news', 'Russell 2000 stocks', 'Russell 2000 futures', 'Russell 2000 shares'], 0)
    nikkei225 = Node('nikkei225', ['Nikkei 225', 'Nikkei 225 index', '"Nikkei 225"', 'Nikkei 225 news', 'Nikkei 225 stocks', 'Nikkei 225 futures', 'Nikkei 225 shares'], 0)
    index = Node('index', ['index stocks', 'world indices', 'index futures', 'company index', 'index shares', 'index news', 'index finance'],
                 0, {nyse_composite, s_p500, nasdaq100, russell2000, nikkei225})

    rub = Node('rub', ['RUB', 'Russian Rouble', '"Russian Rouble"', 'Russian Rouble news', 'Russian Rouble stocks', 'Russian Rouble futures', 'Russian Rouble shares', 'Russian Rouble exchange'], 0)
    gbp = Node('gbp', ['GBP', 'Pound sterling', '"Pound sterling"', 'Pound sterling news', 'Pound sterling stocks', 'Pound sterling futures', 'Pound sterling shares', 'Pound sterling exchange'], 0)
    jpy = Node('jpy', ['JPY', 'Japanese yen', '"Japanese yen"', 'Japanese yen news', 'Japanese yen stocks', 'Japanese yen futures', 'Japanese yen shares', 'Japanese yen exchange'], 0)
    euro = Node('euro', ['EUR', 'Euro', '"Euro"', 'Euro currency news', 'Euro currency stocks', 'Euro currency futures', 'Euro currency shares', 'Euro currency exchange'], 0)
    cny = Node('cny', ['CNY', 'Renminbi', '"Renminbi"', 'Renminbi news', 'Renminbi stocks', 'Renminbi futures', 'Renminbi shares', 'Renminbi exchange'], 0)
    forex = Node('forex', ['forex', '"forex"', 'foreign exchange', 'forex news', 'foreign exchange news', 'forex stocks', 'forex futures', 'forex shares'],
                 0, {rub, gbp, jpy, euro, cny})

    finance = Node('finance', children_set={agriculture, cryptocurrency, metals, energy, index, forex})
    
    if pipeline_folder_path:
        load_node_structure_models(finance, pipeline_folder_path)
    
    return finance
    