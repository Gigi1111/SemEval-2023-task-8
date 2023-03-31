import spacy
from spacy import displacy
from spacy.tokens import Span
from RedditPost import RedditPost

'''
Visualize a reddit post with highlighted spans.
INPUT: A RedditPost object.
OUTPUT: None, it serves an html with the result on your default port on localhost.
'''
def visualise_reddit_post(post):
    #Define tokenizer and doc object
    nlp = spacy.blank("en")
    doc = nlp(post.text)

    spans = []
    '''
    Loop through a post labeled spans and define the custom Spacy doc spans:
    '''
    for a in post.arguments:
      #NOTE: The alignment_mode = contract specifies that only the tokens fully contained in the range defined by the
      #characters will be highlighted. Other options are 'strict' -> the offsets must be found on a token boundary,
      # and 'expand'-> highlights token that are also only partially covered by the range.
      span = doc.char_span(a['start_offset'],a['end_offset'], label=a['label'], alignment_mode = 'expand')
      spans.append(span)
    
    doc.set_ents(spans)
    doc.user_data["title"] = "Post ID: "+post.post_id
    #Customization and definition of labels
    colors = {"question": "#76b5c5", "per_exp": "#91c4a7", "claim": "#91b8c7", "claim_per_exp": "#9cc151" }
    options = {"ents": ["question","per_exp","claim","claim_per_exp"], "colors": colors}

    #Serve html: 
    displacy.serve(doc, style="ent", options = options)
