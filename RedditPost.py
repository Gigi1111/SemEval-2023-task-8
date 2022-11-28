class RedditPost:
  
  def __init__(self, post_id, subreddit_id, arguments, text):
    self.post_id = post_id
    self.subreddit_id = subreddit_id
    self.text = text
    self.arguments = arguments
  
  def __str__(self):
    return "POST_ID: "+ self.post_id +'\n'+"TEXT: "+self.text + '\n'+ "ARGUMENTS: "+ str(self.arguments)
