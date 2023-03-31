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
import model as md
import spacy
from spacy import displacy
from spacy.tokens import Span
import embeddings as emb
import torch
from transformers import BertTokenizer, BertModel
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, ConfusionMatrix


#Read clean data file: 
file_name = os.path.join(__location__, "claim_per_exp.csv")
data = pd.read_table(file_name ,sep=',')

#Read posts:
posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

#Read post and their constituents (function already run, the constituents were saved in a pt file)
post_spans = torch.load('post_spans.pt')

#Function to read row from table obtained from external csv
def read_ro(row):
    text = row['text']
    span = row['span']
    label = 1
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
    

#X_g = emb.generate_embeddings_binary_classifier(data_points)
#torch.save(X_g, "X_claim_per_exp.pt")

X = torch.load(os.path.join(__location__, "X_claim_per_exp.pt"))
print(len(X))

#THE MODEL:

#Construct label tensor
Y = []
for d in data_points:
    Y.append(d[2])

Y = torch.FloatTensor(Y)
Y = Y.view(len(Y),1)


#Generate test sets
X_test = X[81277:]
Y_test = Y[81277:].view(len(Y[81277:]))

#TRAINING
print("Start training")

criterion = torch.nn.BCELoss()
model = md.Linear(768,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X = X[:81277]
Y = Y[:81277]
num_epochs = 20
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
torch.save(model.state_dict(), os.path.join(__location__, 'model_bin_cep_20ep'))
  
#EVALUATION

count = 0
for y in Y_test:
    if y == 1:
        count += 1
print("Num 1's in test set: " + str(count))
model = md.Linear(768,1)
model.load_state_dict(torch.load(os.path.join(__location__, 'model_bin_cep_20ep')))
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

