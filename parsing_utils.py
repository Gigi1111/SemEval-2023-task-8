import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree


#Global pipeline needs to be declared and passed as argument to this function!
#nlp = st.Pipeline('en', verbose = False)
def get_post_constituency(post_text, nlp):
    """ GET the parse trees of post sentences (singular post)
    Input: RedditPost text?
    Output: List[ParseTree] - 
    """

    doc = nlp(post_text)

    constituencies = [] # hold sentence trees in here

    #get the constituencies
    for sentence in doc.sentences:
        constituencies.append(sentence.constituency)

    #To pretty print a tree, lets say the tree of the first sentence:
    #parsetree = Tree.fromstring(str(constituencies[0]))
    #parsetree.pretty_print()
    
    return constituencies

def get_posts_constituencies(post_texts,nlp):
    """ GET the parse trees of post sentences (all post)
    Input: RedditPost texts (all)
    Output: List [List[ParseTree]] - there are as many inner lists as are sentences in a post.  
    The outer list just holds all the posts constituencies
    """
    constituencies_all = []

    for post_text in post_texts:
        constituencies_all.append(get_post_constituency(post_text,nlp))

    return constituencies_all


#Recursive function to get the text corresponding to a node 
# A global list needs to be passed to the recursive function: superList = []
def parse_node(node, superList):
    """ GET all the nodes and their text for a sentence.
    Output: List[{label,leaves}]   #A node would represent a sentence
    """
    d = dict()
    d["label"] = node.label
    d["leaves"] = []
    if len(node.children) >=1:
        for c in node.children:
            d["leaves"].extend(parse_node(c, superList))
        superList.append(d)
    else:
        d["leaves"].append(node.label)
    
    return d["leaves"]











