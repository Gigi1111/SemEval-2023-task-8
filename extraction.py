import pandas as pd
import json as JSON 
from RedditPost import RedditPost

'''
Convert a data point into a RedditPost class instance
'''
def read_row(row):
    post_id = row['post_id']
    subreddit_id = row['subreddit_id']
    labels  = row['stage1_labels']
    text = row['text']
    
    '''Process labels:'''
    #Extract JSON:
    labels_json = JSON.loads(labels)   
    #Get list of labels
    entities = labels_json[0]['crowd-entity-annotation']['entities']
    arguments = []
    for entity in entities:

        arguments.append(
                        dict( label = entity['label'], 
                              start_offset = entity['startOffset'],
                              end_offset = entity['endOffset'])
                        )

    reddit_post = RedditPost(post_id,subreddit_id,arguments,text)
    return reddit_post


'''
Reads a train with text csv into a list of RedditPost objects.
INPUT: The name of the file
OUTPUT: List[RedditPost()]
'''
def read_posts(file_name):

    data = pd.read_table(file_name ,sep=',')
    #For each row, apply the read_row function to convert the row into a RedditPost object.
    reddit_posts = data.apply(lambda row: read_row(row), axis=1)

    return reddit_posts
