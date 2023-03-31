import os
#needed to acces folder files
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import sys
#  path[0] is reserved for script path 
#needed to acces global files
sys.path.insert(1, os.getcwd())

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
import embeddings as emb
import torch
from transformers import BertTokenizer, BertModel
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, ConfusionMatrix


#Read clean data file: 
file_name = os.path.join(__location__, "clean_questions.csv")
data = pd.read_table(file_name ,sep=',')

#Read posts:
posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

#Read post and their constituents (function already run, the constituents were saved in a pt file)
post_spans = torch.load('post_spans.pt')

#Function to read row from table obtained from external csv
def read_ro(row):
    text = row['text']
    span = row['span']
    label = row['is question']
    return [text, span, label]

X = data.apply(lambda row: read_ro(row), axis=1)
count = 0
data_points = []
for x in X:
    text_key = x[0]
    q_span = x[1]
    if text_key in post_spans:
      count += 1
      constituents = post_spans[text_key]
      data_points.append([text_key,q_span,1])
      for sent in constituents:
          for c in sent: 
              co = ' '.join(c["leaves"])
              if (co != q_span):
                data_points.append([text_key,co,0])
    


#THE MODEL:

#Post embeddings dictionary: 
embeds_posts = dict()
#Running the embeddings might take a bit of time so the post text embeddings were run and saved in "post_embeddings.pt"
#(origin: embeddings.py, "embed_posts()")
em = torch.load("post_embeddings.pt")
for i in range(len(posts)-1):
    embeds_posts[posts[i].text] = em[i]

#Construct label tensor
Y = []
for d in data_points:
    Y.append(d[2])

Y = torch.FloatTensor(Y)
Y = Y.view(len(Y),1)


X = torch.load(os.path.join(__location__, "X_question.pt"))


#TRAINING
print("Start training")

criterion = torch.nn.BCELoss()
model = model.Linear(768,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X = X[:210000]
Y = Y[:210000]
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(len(X)): 
        output = model(X[i])
        
        loss = criterion(output, Y[i])
        #print("Item "+ str(i) + " output: " + str(output)+", loss: "+str(loss.item()))

        loss.backward()
        optimizer.step()
        model.zero_grad()
        running_loss += loss.item()

    average_train_loss = running_loss/ len(X)
    print('Epoch [{}/{}], Train Loss per Epoch: {:.4f}'.format(epoch+1, num_epochs, average_train_loss))


torch.save(model.state_dict(), 'model_bin_q_10ep')

    
#EVALUATION
#Generate test sets
X_test = X[210000:]
Y_test = Y[210000:].view(len(Y[210000:]))

count = 0
for y in Y_test:
    if y == 1:
        count += 1
print("Num 1's in test set: " + str(count))

model = model.Linear(768,1)
model.load_state_dict(torch.load(os.path.join(__location__, 'model_bin_q_10ep')))
model.eval()

preds = []
for x in X_test:
    preds.append(model(x).item())
preds = torch.Tensor(preds)

target = Y_test

metric_f1 = BinaryF1Score(threshold = 0.5)
metric_acc = BinaryAccuracy(threshold = 0.5)
confmat = ConfusionMatrix(task="binary", num_classes=2)

f1 = metric_f1(preds, target)
acc = metric_acc(preds, target)
conf = confmat(preds, target)

print("Accuracy: "+ str(acc.item()))
print("F1 Score: "+ str(f1.item()))
print("Confusion Matrix: ")
print(conf)

