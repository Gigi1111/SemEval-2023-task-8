import parsing_utils as pu
import torch
import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree
import parsing_utils as pu
import torch
from datetime import datetime


def show_what_is_extracted_from_tree():
    #Show what is being extracted from a tree with 1 sentence from post 604
    cons = torch.load( 'constituents.pt')

    c  = cons[604]
    # print(len(c))
    # parsetree1 = Tree.fromstring(str(c[0]))
    # parsetree1.pretty_print()
    parsetree2 = Tree.fromstring(str(c[1]))
    parsetree2.pretty_print()
    # parsetree3 = Tree.fromstring(str(c[2]))
    # parsetree3.pretty_print()


    new_constituency_spans = torch.load('new_constituency_spans.pt')
    cf = new_constituency_spans[604][1]
    for d in cf:
         print(d)