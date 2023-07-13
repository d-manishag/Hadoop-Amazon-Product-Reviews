# Hadoop Amazon Product Reviews - on Google Cloud Platform

Hadoop MapReduce used to gain insightful information from Amazon product reviews. This project is based on the sentiment of the sentences mentioning the product features. Thus it leverage the use of NLP for sentiment analysis and the distributed file systems for big data processing

<img width="1029" alt="Screen Shot 2023-07-12 at 10 52 21 PM" src="https://github.com/d-manishag/Hadoop-Amazon-Product-Reviews/assets/138808132/7b980278-8567-4071-826d-e819621419dd">

## Requirements
- Python 3.6 or higher
- Hadoop 2.7 or higher: For Hadoop Cluster 
- Libraries in Python: nltk, textblob


## Google Cloud - Dataproc
- You can use Dataproc to create one or more Compute Engine instances that can connect to a Cloud Bigtable instance and run Hadoop jobs. This page explains how to use Dataproc to automate the tasks:
- https://cloud.google.com/bigtable/docs/creating-hadoop-cluster


## Product Reviews - NLTK Sentiment Analysis

1.  **Data Cleaning**: Data is cleaned and processed for analysis and the text is cleaned removing punctuations, converting to lowercase etc.

2. **Subjectivity Analysis**: For sentiment analysis subjectivity is crucial to identify if the sentence is of positive or negative sentiment.
TextBlob is used for this purpose.

3. **POS Tagging**: The scripts then use the Natural Language Toolkit (NLTK) to tag parts of speech (POS). This aids in classifying the words in the sentences according to their grammatical function, such as whether they are nouns, adjectives, verbs, etc.

4. **Pattern Extraction**: The programs then extract pertinent patterns from the tagged phrases after they have been annotated. These patterns, which are made up of particular POS tag sequences, can be used to locate remarks on product attributes.

5. **Feature Extraction**: The scripts then list frequent attributes cited in the evaluations, including "storage," "price," "battery," "camera," "screen," "performance," and "quality." This is accomplished by determining if the extracted patterns contain certain common traits.

6. **Sentiment Analysis**: Using the Sentiment Intensity Analyzer from NLTK, the scripts conduct sentiment analysis on the pertinent patterns. This aids in determining whether each comment on a feature is positive, negative, or neutral in attitude.

7. **Results**: The scripts then compile the outcomes for each product. By examining the sentiment polarity of the comments made about each feature, they may determine the benefits and drawbacks of each product. The scripts produce the benefits, drawbacks, and key characteristics of each product.


## Hadoop MapReduce

1. **Set Up Your Environment**: Ensure that you have a Hadoop cluster set up and that you have installed Python 3 and the required Python libraries on all nodes of the cluster.

a) Configuring Multi Node Hadoop Cluster on Google Cloud
- Use Dataproc to create three computer instances
- To run Hadoop jobs
  
b) In GCP Console
- Create bucket for cloud storage

c) Setting up cluster 
- Enable component gateway
- One Master computer and two worker nodes

3. **Download NLTK Data**: Download the necessary NLTK data using the `nltk.download('punkt')` command in Python. This needs to be done on all nodes of the cluster.

4. **Move the Scripts**: Move the mapper.py and reducer.py scripts to your Hadoop cluster. These are the scripts that your MapReduce job will use.
- Mapper
Reading customer reviews for Amazon products
Cleaning the review text by converting it to lowercase and removing punctuation
Filtering out subjective sentences using TextBlob
Performing part-of-speech tagging on the filtered sentences using NLTK
Outputting the ASIN and tagged text as key-value pairs

- Reducer
Taking the output generated by the mapper as input
Identifying relevant patterns in the tagged text using part-of-speech patterns
Extracting features from the relevant patterns by counting their frequency
Performing sentiment analysis on the patterns using NLTK's SentimentIntensityAnalyzer
Outputting the sentiment scores for the top features of each product as a formatted string


5. **Run Your Job**: Run your Hadoop streaming job with mapper.py as the mapper script and reducer.py as the reducer script. The scripts will read your input data, process it, and write the output to your specified output directory.

## HIVE 
Hive is built on top of Apache Hadoop, which is an open-source framework used to efficiently store and process large datasets. As a result, Hive is closely integrated with Hadoop, and is designed to work quickly on petabytes of data.
Here HIVE helps perform ETL functionality helping to extract insights!


![Picture1](https://github.com/d-manishag/Hadoop-Amazon-Product-Reviews/assets/138808132/cc707c1d-dc06-41c6-9d08-314aef521782)
![Picture2](https://github.com/d-manishag/Hadoop-Amazon-Product-Reviews/assets/138808132/1a4e877a-0e8a-46e6-a18d-b7bff5223572)
![Picture3](https://github.com/d-manishag/Hadoop-Amazon-Product-Reviews/assets/138808132/1bb4f932-3de1-4d46-8e07-3c243915cc99)


![Picture4](https://github.com/d-manishag/Hadoop-Amazon-Product-Reviews/assets/138808132/2f9e9123-4926-4329-97ec-6f2468351028)


