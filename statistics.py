#CODE NOT RELEVANT JUST ME LOOKING INTO THE DATA 
import torch
import transformers
from transformers import BertTokenizer, BertModel
import embeddings as t
from dataset_generation import convert_posts_to_docs
import model
import extraction as ex
import matplotlib.pyplot as plt
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English




















def old_stuff_that_do_not_want_to_delete_yet():
        nlp = English()
        # Create a blank Tokenizer with just the English vocab
        tokenizer = Tokenizer(nlp.vocab)

        posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')
        docs = convert_posts_to_docs(posts)
        x= []
        for doc in docs: 
                post_text = str(doc) #the text of the post : String
                spans = doc.ents     #the lists of spans: List[String]
                x.append(len(tokenizer(str(post_text))))
                #the span
               # for span in spans:
        
        #create list of data

        #create histogram with 4 bins
        plt.hist(x, bins=10, edgecolor='black')
        plt.show()