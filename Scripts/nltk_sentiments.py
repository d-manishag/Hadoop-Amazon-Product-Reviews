import json
import string
import nltk
from textblob import TextBlob
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer

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

def mapper_reducer(input_file_path, output_file_path):
    # Read the input data from a file
    with open(input_file_path, 'r') as f:
        input_data = f.readlines()

    # Mapper script
    intermediate_results = []
    for line in input_data:
        try:
            review = json.loads(line)
            review_text = clean_text(review['reviewText'])
            subjective_sentences = filter_subjective_sentences(review_text)
            tagged_text = pos_tagging(" ".join(subjective_sentences))
            intermediate_results.append((review['asin'], tagged_text))
        except Exception as e:
            continue

    # Reducer script
    current_asin = None
    tagged_texts = []
    results = {}

    for asin, tagged_text in intermediate_results:
        if current_asin == asin:
            tagged_texts.append(tagged_text)
        else:
            if current_asin:
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

                results[current_asin] = {
                    'pros': pros,
                    'cons': cons,
                    'top_features': sorted_features
                }

            current_asin = asin
            tagged_texts = [tagged_text]

    # Processing the last group of tagged_texts
    if current_asin:
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

        results[current_asin] = {
            'pros': pros,
            'cons': cons,
            'top_features': sorted_features
        }

    # Write the results to the output file
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(results, indent=4))

# Run the combined script on the input data
mapper_reducer("input_file.json", "output_file.json")
