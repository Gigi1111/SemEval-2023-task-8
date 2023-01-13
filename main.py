import visualisation_utils as vs
import extraction as ex

#Read posts
posts = ex.read_posts('st1_public_data/st1_train_inc_text.csv')

#Get a reddit post, lets say the second one: 
post = posts[106]
post_text = post.text
print(post_text)
#Visualise it: 
vs.visualise_reddit_post(post)

