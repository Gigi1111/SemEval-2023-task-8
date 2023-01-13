#GET SENTENCE EMBEDDINGS (well used for either )
import torch
from transformers import BertTokenizer, BertModel
from datetime import datetime


def get_sentence_embedding(input_string, tokenizer, emb_model):
    e = tokenizer.encode(input_string,truncation = True, return_tensors = 'pt')
    emb_model.eval()
    with torch.no_grad():
        outputs = emb_model(e)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1) #if 0 remains unchanged
    token_embeddings = token_embeddings.permute(1,0,2)

    #SENTENCE EMBEDDINGS:
    token_vecs = hidden_states[-2][0]#what average, this gets only the second last hidden layer
    # Calculate the average of all 22 token vectors.
    sentence_embeddings = torch.mean(token_vecs, dim=0)

    return sentence_embeddings  #[768]

def construct_input(spans, tokenizer, emb_model): # posts = X post, span, label
    X = []
    i = 1 
    for span in spans: 
        #the text of the post where the span is coming from 
        #post_text_emb = get_sentence_embedding(span[0], tokenizer, emb_model) 
        #the span
        span_text_emb = get_sentence_embedding(span[1], tokenizer, emb_model)
        #X.append(torch.cat((post_text_emb,span_text_emb)))
        X.append(span_text_emb)
        
        i += 1
        if i % 100 == 0:
            print("Data point "+ str(i)+ " done")
    return X

#the more efficeint method, same as result as function above just more efficient time wise
def construct_input_eff(docs, tokenizer, emb_model): # posts = X post, span, label
    X = []
    i = 1 
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    for doc in docs: 
        post_text = str(doc) #the text of the post : String
        spans = doc.ents     #teh lists of spans: List[String]
        #the text of the post where the span is coming from 
        post_text_emb = get_sentence_embedding(post_text, tokenizer, emb_model) 
        #the span
        for span in spans:
            span_text_emb = get_sentence_embedding(str(span), tokenizer, emb_model)
        
            X.append(post_text_emb + span_text_emb)
        
        i += 1
        if i % 100 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Data point "+ str(i)+ " done at "+ current_time,)
    return X
        
