from cProfile import label
from types import NoneType
import extraction as ex
import spacy


def convert_posts_to_docs(posts):

    #Read all posts from the training file
    
    nlp = spacy.blank("en")

    #Create the dataset:
    dataset = []#list of docs 

    #Convert each RedditPost object into spacy Doc object - easier to extract and manage spans
    for post in posts:  

        spans = [] #where the spans of each post will be collected
        doc = None 
        doc = nlp(post.text) #Initialize a new Doc object with the text of the reddit post  
        for a in post.arguments: #extract spans using spacy's Doc functions, using the offsets
              #NOTE: The alignment_mode = contract specifies that only the tokens fully contained in the range defined by the
              #characters will be highlighted. Other options are 'strict' -> the offsets must be found on a token boundary,
              # and 'expand'-> highlights token that are also only partially covered by the range.
              span = doc.char_span(a['start_offset'],a['end_offset'], label=a['label'], alignment_mode = 'contract')
              if span is not None:
               spans.append(span) #collect all of the posts' spans here

        doc.set_ents(spans) #attach the spans to the Doc 
        dataset.append(doc) #collect the Docs here
    return dataset

    #print(dataset[1].ents[0].label_) #get label of the first span of the first post 



def generate_dataset(docs):
    #dataset
    X = []
    #generate dataset:

    keys = ["question","per_exp","claim","claim_per_exp"]
    dataset = docs
    for doc_post in dataset:

        post_text = str(doc_post) #the text of the post : String
        spans = doc_post.ents     #teh lists of spans: List[String]

        for span in spans:
            #initialize dict, mark all labels with '0':
            labels = dict.fromkeys(keys, 0)
            labels[span.label_] = 1 #change the value of the corresponding label to 1
            data_point = [post_text,str(span),list(labels.values())] #how the datapoint looks
            X.append(data_point) #the grain here is the span and it has the text and one hot encoding of labels

    return X  # X[0] = ['some redit post..', 'a span..', [1,0,0,0]] X= [[],[],[]...] - datapoints like X[0]
    #print(len(X)) #12168

