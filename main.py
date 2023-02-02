import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree
import parsing_utils as pu
import torch
from datetime import datetime


#Read posts for training
#posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

# #Get all constituencies 

# texts = []
# for post in posts: 
#   texts.append(post.text)

# print("Get constituents now: ")
# nlp = st.Pipeline('en', verbose = False)
# cons = pu.get_posts_constituencies(texts,nlp)
# torch.save(cons, 'constituents.pt')


#Load constituencies
#constituency_spans = torch.load( 'constituency_spans.pt')

#Load filtered constituencies
#new_constituency_spans = pu.filter_constituency_spans(constituency_spans)



