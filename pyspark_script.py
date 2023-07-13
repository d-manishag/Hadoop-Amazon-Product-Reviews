from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import json
import string
import nltk
from textblob import TextBlob
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

def clean_text(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def filter_subjective_sentences(text):
    blob = TextBlob(text)
    return [str(sentence) for sentence in blob.sentences if sentence.sentiment.subjectivity >= 0.3]

def pos_tagging(text):
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)

def extract_relevant_patterns(tagged_text):
    patterns = [('RB', 'JJ', 'NN'), ('JJ', 'NN'), ('RB', 'VB', 'NN')]
    relevant_patterns = []

    for pattern in patterns:
        for i in range(len(tagged_text) - len(pattern) + 1):
            if tuple(tag[1] for tag in tagged_text[i:i + len(pattern)]) == pattern:
                relevant_patterns.append(tuple(tag[0] for tag in tagged_text[i:i + len(pattern)]))

    return relevant_patterns

def extract_features(relevant_patterns):
    feature_freq = defaultdict(list)
    for pattern in relevant_patterns:
        feature = pattern[-1]
        feature_freq[feature].append(pattern)

    common_words = ['storage', 'price', 'battery', 'camera', 'screen', 'performance', 'quality']
    features = {word: freq for word, freq in feature_freq.items() if word in common_words}
    return features

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def process_review(review_json):
    review = json.loads(review_json)
    review_text = clean_text(review['reviewText'])
    subjective_sentences = filter_subjective_sentences(review_text)
    tagged_text = pos_tagging(" ".join(subjective_sentences))
    return (review['asin'], tagged_text)

def process_asin(asin, tagged_texts):
    all_relevant_patterns = []
    for tt in tagged_texts:
        all_relevant_patterns += extract_relevant_patterns(tt)
    features = extract_features(all_relevant_patterns)

    overall_polarity = {feature: 0 for feature in features}
    for feature in features:
        relevant_patterns = features[feature]
        for pattern in relevant_patterns:
            sentiment = sentiment_analysis(" ".join(pattern))
            overall_polarity[feature] += sentiment
        overall_polarity[feature] /= len(relevant_patterns)

    pros = [feature for feature, score in overall_polarity.items() if score >= 0]
    cons = [feature for feature, score in overall_polarity.items() if score < 0]
    sorted_features = sorted(overall_polarity.items(), key=lambda x: x[1], reverse=True)

    return (asin, {
        'pros': pros,
        'cons': cons,
        'top_features': sorted_features
    })

# Set up the SparkContext
conf = SparkConf().setAppName("ProductReviewAnalyzer")
sc = SparkContext(conf=conf)

spark = SparkSession(sc)



reviews = sc.textFile("input_data.json")

# Process the reviews
processed_reviews = reviews.map(process_review)

# Group the tagged texts by ASIN
grouped_reviews = processed_reviews.groupByKey()

# Process each ASIN
results = grouped_reviews.map(lambda x: process_asin(x[0], list(x[1])))


# Convert the results to a DataFrame
results_df = spark.createDataFrame(results)

# Write the DataFrame to a JSON file
results_df.write.json("output.json")
