import sys
import json
import string
import nltk
nltk.data.path.append('/home/vc508/nltk_data')
from textblob import TextBlob

def clean_text(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def filter_subjective_sentences(text):
    blob = TextBlob(text)
    return [str(sentence) for sentence in blob.sentences if sentence.sentiment.subjectivity >= 0.3]

def pos_tagging(text):
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)

for line in sys.stdin:
    try:
        review = json.loads(line)
        review_text = clean_text(review['reviewText'])
        subjective_sentences = filter_subjective_sentences(review_text)
        tagged_text = pos_tagging(" ".join(subjective_sentences))
        #print(review['asin'], json.dumps(tagged_text))
        print(f"{review['asin']}\t{json.dumps(tagged_text)}")
    except Exception as e:
        continue
