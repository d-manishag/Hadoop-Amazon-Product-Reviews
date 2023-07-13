import sys
import json
import nltk
nltk.data.path.append('/home/vc508/nltk_data')
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer

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

current_asin = None
tagged_texts = []
results = {}

for line in sys.stdin:
    asin, tagged_text_json = line.strip().split('\t', 1)
    tagged_text = json.loads(tagged_text_json)

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

#Output the results
print("ASIN, storage, price, battery, camera, screen, performance, quality")
for asin, result in results.items():
    feature_polarities = {feature: polarity for feature, polarity in result['top_features']}
    formatted_result = [
        f"{asin}",
        f"{feature_polarities.get('storage', 'N/A'):.2f}" if 'storage' in feature_polarities else "N/A",
        f"{feature_polarities.get('price', 'N/A'):.2f}" if 'price' in feature_polarities else "N/A",
        f"{feature_polarities.get('battery', 'N/A'):.2f}" if 'battery' in feature_polarities else "N/A",
        f"{feature_polarities.get('camera', 'N/A'):.2f}" if 'camera' in feature_polarities else "N/A",
        f"{feature_polarities.get('screen', 'N/A'):.2f}" if 'screen' in feature_polarities else "N/A",
        f"{feature_polarities.get('performance', 'N/A'):.2f}" if 'performance' in feature_polarities else "N/A",
        f"{feature_polarities.get('quality', 'N/A'):.2f}" if 'quality' in feature_polarities else "N/A"
    ]
    print(", ".join(formatted_result))

