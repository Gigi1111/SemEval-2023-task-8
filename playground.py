#FILE NOT RELEVANT JUST ME PLAYING AROUND AND TESTING THINGS
import torch
import transformers
from transformers import BertTokenizer, BertModel
import embeddings as t
from dataset_generation import convert_posts_to_docs
import model
import extraction as ex

#posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

#docs = convert_posts_to_docs(posts)

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased',
#                                  output_hidden_states = True, # Whether the model returns all hidden-states.
#                                 )


#X = t.construct_input_eff(docs, tokenizer, model)
#torch.save(X, 'file.pt')

c = torch.load('file.pt')
print(len(c))
inp = c[:5]
m = model.Linear(768,4)
out = m(inp[0])
print(out)





posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')
docs = convert_posts_to_docs(posts)

X = torch.load('file.pt')
dataset = pd.DataFrame(generate_dataset(docs),columns = ['post','span','digits'])
Y = torch.Tensor(list(dataset['digits']))