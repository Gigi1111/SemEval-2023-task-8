# SemEval-2023-task-8

Scrape Data from Reddit
Step 1. refer to this Medium article
https://towardsdatascience.com/scraping-reddit-data-1c0af3040768

Step 2. sign in and create a Reddit app
https://www.reddit.com/prefs/apps

Step 3. fill in the "xxx" part with your Reddit app information and start scraping
python3 extraction_script.py --client_id xxx --user_agent xxx --client_secret xxx --username xxx --password xxx --file_path sample/st2_train_sample.csv


## Project description 

This repository was developed as part of the CogSys Master's project module at the University of Potsdam, Germany. Full paper of the project can be found [here](https://www.overleaf.com/project/63737db61470cc4405a391c3). Which follows the SemEval System paper [Structure](https://semeval.github.io/system-paper-template.html) and [Requirements](https://semeval.github.io/paper-requirements.html).

The goal is to come up with a system that fulfills the requirements of [this task](https://causalclaims.github.io/).


### Subtask 1: Causal claim identification:
For the provided snippet of text, the first subtask aims to identify the span of text that is either a claim, experience, experience_based_claim or a question. These four categories can be defined as follow:

Claim: Commmunicates a causal interaction between an intervention and an outcome.
Experience: Relates a specific outcome/symptom to an intervention or population based on someone's experience.
Experience based claim: A claim based on someone's experince.
Question: Poses a question.
Participants can work on it at sentence level and try to classify sentences in one of the given classes but many times claim (or other class) is just a part of the sentence. But this maybe only a baseline as in many examples only a part of sentence is annotated as one of these category. Please check the image below for more clarity.


### Subtask 2: PIO frame extraction:
In this subtask, for a given multi (or single) sentence text snippet and identified claim in that snippet, the task is to extract related Population (P), Intervention (I), and Outcome (O) frame. While it is rare, it may be the case that there is more than one claim in any given post. In that case, we want to identify PIO elements for a given claim. This can be framed as a sequence tagging task.


## Corpora 

The Reddit texts were extracted from the Codalab folder by the SemEval task 8 creators [here](https://codalab.lisn.upsaclay.fr/competitions/6948?secret_key=0eb18fd8-c847-4738-956c-f0f19fe3692e#participate-get_starting_kit). 

### Subtask 1
For subtask 1, the training data contains 5695 reddit texts, and from which, we got 127359 sentences with their corresponding label on the sentence level.

