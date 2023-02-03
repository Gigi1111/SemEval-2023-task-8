import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree
import parsing_utils as pu
import torch
import torch.nn as nn
from datetime import datetime
import model

#Take an example of a post: 
#Read posts for training
posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

post = posts[700]


filtered_spans = torch.load('new_constituency_spans.pt')

print(post.text)

#Visualize it: 
#vs.visualise_reddit_post(post)

#the tree:
cons = torch.load('constituents.pt')


#To pretty print a tree, lets say the tree of the first sentence:

parsetree = Tree.fromstring(str(cons[700][0]))
parsetree.pretty_print()

#All iltered candidates (for first sentence.)
c_spans = filtered_spans[700]
c_spans_0 = c_spans[0]

for e in c_spans_0:
    print(e)




#Prepare input:


