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
import spacy
from spacy import displacy
from spacy.tokens import Span
from RedditPost import RedditPost
import embeddings as em
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import parsing_utils as pu

nm = pu.create_post_constituents_dictionary()

# x = torch.load("X.pt")
# pred = 
# label = Y[]
# model = model.Linear(768,1)
# model.load_state_dict(torch.load('model_1'))
# model.eval()
# # print(model(x[3474]))

# target = torch.tensor([0, 1, 0, 1, 0, 1])
# preds = torch.tensor([0, 0.6, 0.3, 0.6, 0, 0.4])
# metric = BinaryF1Score(threshold = 0.5)
# metric_acc = BinaryAccuracy(threshold = 0.5)
# f1 = metric(preds, target)
# acc = metric_acc(preds, target)
# print(acc)


# em = torch.load("post_embeddings.pt")

#Read posts for training
#posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')
# vs.visualise_reddit_post(posts[500])
# filtered_spans = torch.load('new_constituency_spans.pt')
# for e in filtered_spans[1][1]:
#     print(e)
# #the tree:
# nlp = st.Pipeline('en', verbose = False)
# cc = pu.get_post_constituency(posts[500].text, nlp)

# #To pretty print a tree, lets say the tree of the first sentence:

# parsetree = Tree.fromstring(str(cc[1]))
# parsetree.pretty_print()


################
# consts = []
# post = posts[640]
# spans = []
# nlp = st.Pipeline('en', verbose = False)
# doc = nlp(post.text)
# '''
# Loop through a post labeled spans and define the custom Spacy doc spans:
# '''
# for a in post.arguments:
#   #NOTE: The alignment_mode = contract specifies that only the tokens fully contained in the range defined by the
#   #characters will be highlighted. Other options are 'strict' -> the offsets must be found on a token boundary,
#   # and 'expand'-> highlights token that are also only partially covered by the range.
#   span = doc.char_span(a['start_offset'],a['end_offset'], label=a['label'], alignment_mode = 'expand')
#   spans.append(span) 
#   consts.append(pu.get_string_constituency(str(span),nlp))
   
# doc.set_ents(spans)

# parsetree = Tree.fromstring(str(consts[1]))
# parsetree.pretty_print()
################

# #the text of the post where the span is coming from 
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased',
#                                   output_hidden_states = True, # Whether the model returns all hidden-states.
#                                   )
# l = len(data_points) 
# count = 0                          
# for x in data_points:
#     count += 1
#     if(count%100 ==0):
#       print(str(count)+" /"+str(l))
#     text_key = x[0]
#     if text_key in embeds_posts: 
#       post_text_emb  = embeds_posts[text_key] #post embedding
#       span_emb = emb.get_sentence_embedding(x[1], tokenizer, model) 
#       X.append(post_text_emb + span_emb)






# #Get all constituencies

# texts = []
# for post in posts: 
#   texts.append(post.text)

# print("Get constituents now: ")
# nlp = st.Pipeline('en', verbose = False)
# cons = pu.get_posts_constituencies(texts,nlp)
# torch.save(cons, 'constituents.pt')


#Load constituencies
#constituency_spans = torch.load('constituency_spans.pt')

#Load filtered constituencies
#new_constituency_spans = pu.filter_constituency_spans(constituency_spans)



# #train model
# X = torch.load('file.pt')
# f = nn.Softmax(dim = 0 )
# model = model.Linear(768,5)
# model.load_state_dict(torch.load('model_1'))
# model.eval()

# o = model(X[1])
# print(f(o))