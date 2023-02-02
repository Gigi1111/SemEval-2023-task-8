import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree
from datetime import datetime
import torch

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
    
    c=1
    for post_text in post_texts:

        constituencies_all.append(get_post_constituency(post_text,nlp))
        print(c)
        c += 1

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


def get_posts_constituency_spans(constituencies_all):
    """ GET the spans of all nodes of post sentences (all posts)
    Input: List [List[ParseTree]] (all)
    Output: List [List[List[{label,leaves}]]] - there are as many inner lists as are sentences in a post.  
    second list - container for posts
    outer list - container for all posts 
    The outer list just holds all the posts constituencies
    """
    cons = constituencies_all
    final = []  
    count = 1
    for post in cons:
        print("Post: "+ str(count))
        post_nodes = []
        co_s = 1
        for node in post:
            print("Sentence "+str(co_s))
            sentence_leaves = []
            parse_node(node, sentence_leaves)
            post_nodes.append(sentence_leaves)
            co_s += 1
        final.append(post_nodes)
        count += 1
        now = datetime.now()
        print("Post: "+ str(count)+" ready at "+ now.strftime("%H:%M:%S"))   
    torch.save(final, 'constituency_spans.pt')
    return final


def filter_constituency_spans(constituency_spans):
    required_nodes = ["S","SBARQ","SQ","SBAR","VP"] #to be potentilly extended
    new_constituency_spans = []
    count = 1
    for post in constituency_spans:
        post_c = []
        for sentence in post:
            sent = []
            for dict in sentence:
                if dict["label"] in required_nodes:
                    sent.append(dict)
            post_c.append(sent)
        new_constituency_spans.append(post_c)
        count += 1
        now = datetime.now()
        print("Post: "+ str(count)+" ready at "+ now.strftime("%H:%M:%S"))   
    torch.save(new_constituency_spans, 'new_constituency_spans.pt')
    return new_constituency_spans





