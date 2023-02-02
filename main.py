import visualisation_utils as vs
import extraction as ex
import pandas as pd
import stanza as st
from nltk.tree import Tree
import parsing_utils as pu
import torch

from datetime import datetime



# #Read posts for training
#posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')


# texts = []
# for post in posts: 

#     texts.append(post.text)


# #Get all constituencies 
# print("Get constituents now: ")
# nlp = st.Pipeline('en', verbose = False)

# cons = pu.get_posts_constituencies(texts,nlp)

# torch.save(cons, 'constituents.pt')

cons = torch.load( 'constituents.pt')

c  = cons[604]
# print(len(c))
# parsetree1 = Tree.fromstring(str(c[0]))
# parsetree1.pretty_print()
parsetree2 = Tree.fromstring(str(c[1]))
parsetree2.pretty_print()
# parsetree3 = Tree.fromstring(str(c[2]))
# parsetree3.pretty_print()







constituency_spans = torch.load( 'constituency_spans.pt')

# p = constituency_spans[604]

# for e in p[1]:
#     print(e)

#Filter the constituents: 

# required_nodes = ["S","SBARQ","SQ","SBAR","VP"] #to be potentilly extended
# new_constituency_spans = []
# count = 1
# for post in constituency_spans[:2]:
#     post = []
#     for sentence in post:
#         sent = []
#         for dict in sentence:
#             if dict["label"] in required_nodes:
#                 sent.append(dict)
#         post.append(sent)
#     new_constituency_spans.append(post)
#     count += 1
#     now = datetime.now()
#     print("Post: "+ str(count)+" ready at "+ now.strftime("%H:%M:%S"))   
# torch.save(new_constituency_spans, 'new_constituency_spans.pt')

#new_constituency_spans = pu.filter_constituency_spans(constituency_spans)

new_constituency_spans = torch.load('new_constituency_spans.pt')
cf = new_constituency_spans[604][1]

for d in cf:
     print(d)