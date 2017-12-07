# LDA
## Using Latent Dirichlet Allocation to analyze customer reviews
	Importing packages
	from collections import defaultdict
	from pyspark import SparkContext
	from pyspark.mllib.linalg import Vector, Vectors
	from pyspark.mllib.clustering import LDA, LDAModel
	from pyspark.sql import SQLContext
	import re

## Defining our thresholds 
	num_of_stop_words = 50      # Number of most common words to remove, trying to eliminate stop words
	num_topics = 3	            # Number of topics we are looking for
	num_words_per_topic = 5    # Number of words to display for each topic
	max_iterations = 35         # Max number of times to iterate before finishing_

## Importing a file containing textual reviews
	data3=sc.wholeTextFiles("s3://s3-ankit/text/Reviews.csv")
## Extracting the text from the tuple into an RDD
	data3=data3.map(lambda line: line[1])
## tokenize the data to form our global vocabulary
	tokens = data3.map( lambda document: document.strip().lower()).map( lambda document: re.split("[\s;,#]", document))       \
	.map( lambda word: [x for x in word if x.isalpha()]).map( lambda word: [x for x in word if len(x) > 3] )

### 1. Put all the words in one list instead of a list per document
### 2. Mapping 1 to each entry
### 3. Reduce the tuples by key, i.e.: Merge all the tuples together by the word, summing up the counts
### 4. Reverse the tuple so that the count to make 
### 5. sort by the word count

	termCounts = tokens                             \
	.flatMap(lambda document: document)         \
	.map(lambda word: (word, 1))                \
	.reduceByKey( lambda x,y: x + y)            \
	.map(lambda tuple: (tuple[1], tuple[0]))    \
	.sortByKey(False)
## Identify a threshold to remove the top words, in an effort to remove stop words

	threshold_value = termCounts.take(num_of_stop_words)[num_of_stop_words - 1][0]
	threshold_value

4
## Retain words with a count less than the threshold identified above, 
## Index each one and collect them into a map

	vocabulary = termCounts                         \
	.filter(lambda x : x[0] < threshold_value)  \
	.map(lambda x: x[1])                        \
	.zipWithIndex()                             \
	.collectAsMap()

## Convert the given document into a vector of word counts

	def document_vector(document):
	id = document[1]
	counts = defaultdict(int)
	for token in document[0]:
	if token in vocabulary:
	token_id = vocabulary[token]
	counts[token_id] += 1
	counts = sorted(counts.items())
	keys = [x[0] for x in counts]
	values = [x[1] for x in counts]
	return (id, Vectors.sparse(len(vocabulary), keys, values))

## Process all of the documents into word vectors using the 

	documents = tokens.zipWithIndex().map(document_vector).map(list)

## Inverting the key value to get index value

	inv_voc = {value: key for (key, value) in vocabulary.items()}

## Print topics, showing the top-weighted 10 terms for each topic

	lda_model = LDA.train(documents, k=num_topics, maxIterations=max_iterations)
	topic_indices = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)

## Print topics, showing the top-weighted 10 terms for each topic
	for i in range(len(topic_indices)):
	print("Topic #{0}\n".format(i + 1))
	for j in range(len(topic_indices[i][0])):
	print("{0}\t{1}\n".format(inv_voc[topic_indices[i][0][j]] \
	.encode('utf-8'), topic_indices[i][1][j]))
	
	print("{0} topics distributed over {1} documents and {2} unique words\n" .format(num_topics, documents.count(), len(vocabulary)))

Topic #1

seemed	0.00505424295953

best	0.0049822157329

been	0.00493771102385

crew	0.00491934921165

think	0.00490713449392

Topic #2

mexico	0.00494354351306

vacation	0.00491706033678

took	0.00488545807175

also	0.00487481475589

lots	0.00485502341206

Topic #3

appreciate	0.00502999395327

early	0.00498803502504

sure	0.00493104622834

ticket	0.00489850995821

plane	0.00488957039609

3 topics distributed over 1 documents and 476 unique words
