import torch
import transformers
from transformers import BertTokenizer, BertModel
import embeddings as t
from dataset_generation import convert_posts_to_docs,generate_dataset
import model
import pandas as pd
import numpy as np
import extraction as ex

#running the embeddings might take a bit of time so I saved the resulting 
# vectors in tensor files
X = torch.load('file.pt')
Y_i= torch.load('labels.pt')

#Add slot for "None" category
z = torch.zeros((len(Y_i),1))
Y_i =torch.cat((Y_i,z),1)

Y = np.argmax(Y_i, axis=1)



#TRAINING

criterion = torch.nn.CrossEntropyLoss()
model = model.Linear(768,5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X = X
Y = Y
num_epochs = 500

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


torch.save(model.state_dict(), 'model_1')


                
#EVALUATION
#TBD


