# Random Forest tutorial

## Introduction
spark.ml is a new package introduced in Spark 1.2, which aims to provide a uniform set of high-level APIs that help users create and tune practical machine learning pipelines. This tutorial will explore SparkML in detail. Our tutorial will cover both an unsupervised and supervised machine learning algorithm from the sparkML library. To better demonstrate the power of Spark and AWS, the project will demonstrate customer segmentation of an airline dataset. To demonstrate the power of Spark and AWS, the project will demonstrate customer segmentation of an airline dataset and also look at analyzing emails to classify spam messages.

## Supervised Learning
Binary classification is a supervised learning problem in which we want to classify entities into one of two distinct categories or labels. This problem involves executing a learning Algorithm on a set of labeled examples, that is , a set of entities represented via features along with underlying category labels. The algorithm returns a trained Model that can predict the label for new entities for which the underlying label is unknown. 
To demonstrate binary classification, we are using data XYZ Corportation which is a International conglomerate operating in many domains and geographies. XYZ employees receive thousands of emails on a daily basis. As an improved security measure XYZ would like implement a custom spam filter for its employee emails and classifiy the received mails into spam and not spam.

## Load and Explore Data
First of all, we will load the csv file which has the labeled data. The data has 57 columns and 4601 rows, where the last column is the label. A label value of 1 signifies that the mail is a spam while a value of 0 signifies its not a spam.

Data Source: https://archive.ics.uci.edu/ml/datasets/spambase

	data = sqlContext.read.format('com.databricks.spark.csv') \
	.option("inferSchema",True).option("header",True).load("s3://s3-ankit/Spam_data/HW4_DATA.csv")
For easy reading, we can convert the dataframe to Pandas dataframe and display the first three rows.
	data.limit(3).toPandas()
 word_freq_make  word_freq_address  word_freq_all  word_freq_3d  \\
0            0.00               0.64           0.64             0   
1            0.21               0.28           0.50             0   
2            0.06               0.00           0.71             0   

   word_freq_our  word_freq_over  word_freq_remove  word_freq_internet  \\
0           0.32            0.00              0.00                0.00   
1           0.14            0.28              0.21                0.07   
2           1.23            0.19              0.19                0.12   

   word_freq_order  word_freq_mail     ...       char_freq_;  char_freq_(  \\
0             0.00            0.00     ...              0.00        0.000   
1             0.00            0.94     ...              0.00        0.132   
2             0.64            0.25     ...              0.01        0.143   

   char_freq_[  char_freq_!  char_freq_$  char_freq_\#  \\
](#)0          0.0        0.778        0.000        0.000   
1          0.0        0.372        0.180        0.048   
2          0.0        0.276        0.184        0.010   

   capital_run_length_avera  capital_run_length_longe  \\
0                     3.756                        61   
1                     5.114                       101   
2                     9.821                       485   

   capital_run_length_total  spam_nospam  
0                       278            1  
1                      1028            1  
2                      2259            1  

[3 rows x 58 columns](#)

To view the schema of the dataframe, we can use printSchema(). This operation displays the schema as a visual tree.
	data.printSchema()

root
 |-- word_freq_make: double (nullable = true)
 |-- word_freq_address: double (nullable = true)
 |-- word_freq_all: double (nullable = true)
 |-- word_freq_3d: integer (nullable = true)
 |-- word_freq_our: double (nullable = true)
 |-- word_freq_over: double (nullable = true)
 |-- word_freq_remove: double (nullable = true)
 |-- word_freq_internet: double (nullable = true)
 |-- word_freq_order: double (nullable = true)
 |-- word_freq_mail: double (nullable = true)
 |-- word_freq_receive: double (nullable = true)
 |-- word_freq_will: double (nullable = true)
 |-- word_freq_people: double (nullable = true)
 |-- word_freq_report: double (nullable = true)
 |-- word_freq_addresses: double (nullable = true)
 |-- word_freq_free: double (nullable = true)
 |-- word_freq_business: double (nullable = true)
 |-- word_freq_email: double (nullable = true)
 |-- word_freq_you: double (nullable = true)
 |-- word_freq_credit: double (nullable = true)
 |-- word_freq_your: double (nullable = true)
 |-- word_freq_font: double (nullable = true)
 |-- word_freq_000: double (nullable = true)
 |-- word_freq_money: double (nullable = true)
 |-- word_freq_hp: double (nullable = true)
 |-- word_freq_hpl: double (nullable = true)
 |-- word_freq_george: integer (nullable = true)
 |-- word_freq_650: double (nullable = true)
 |-- word_freq_lab: double (nullable = true)
 |-- word_freq_labs: integer (nullable = true)
 |-- word_freq_telnet: double (nullable = true)
 |-- word_freq_857: integer (nullable = true)
 |-- word_freq_data: double (nullable = true)
 |-- word_freq_415: double (nullable = true)
 |-- word_freq_85: double (nullable = true)
 |-- word_freq_technology: double (nullable = true)
 |-- word_freq_1999: double (nullable = true)
 |-- word_freq_parts: double (nullable = true)
 |-- word_freq_pm: double (nullable = true)
 |-- word_freq_direct: double (nullable = true)
 |-- word_freq_cs: integer (nullable = true)
 |-- word_freq_meeting: double (nullable = true)
 |-- word_freq_original: double (nullable = true)
 |-- word_freq_project: double (nullable = true)
 |-- word_freq_re: double (nullable = true)
 |-- word_freq_edu: double (nullable = true)
 |-- word_freq_table: integer (nullable = true)
 |-- word_freq_conference: double (nullable = true)
 |-- char_freq_;: double (nullable = true)
 |-- char_freq_(: double (nullable = true)
 |-- char_freq_[: double (nullable = true)
](#) |-- char_freq_!: double (nullable = true)
 |-- char_freq_$: double (nullable = true)
 |-- char_freq_\#: double (nullable = true)
 |-- capital_run_length_avera: double (nullable = true)
 |-- capital_run_length_longe: integer (nullable = true)
 |-- capital_run_length_total: integer (nullable = true)
 |-- spam_nospam: integer (nullable = true)
## Train Test Split
Next, we split the data into training and testing set. The training set is data on which our model will be trained, while the testing set is holdout data on which we test the performance of our trained model. The training set will comprise of 70% of the data and the rest 30% will be the testing/holdout data.

	
	train, test = df.randomSplit([0.7, 0.3])
	print "We have %d training examples and %d test examples." % (train.count(), test.count())
## Formatting the Data
We need to format data to bring it in a format which is acceptable to the ML algorithms. 

### StringIndexer
It encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0. We convert the label column to a column of label indices.
](#)
### VectorAssembler
It is a transformer that combines a given list of columns into a single vector column. We use it to combine all the raw features which we plan to use in our model(labeled column is excluded). It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like random forest. 

## Train the Model

Next, we train the model using the RandomForestClassifier(). In this operation, we specify our labeled column(indexedLabel) and the features we plan to use. We also mention the number of trees that we want our model to train on( in our case it is 50).

Next, we create a evaluator where we can specify the metric which we want to evaluate. in our case, we use 'accuracy'.

## Cross-Validation on our Model

In k-fold cross-validation, the original sample is randomly partitioned into k equal size subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged (or otherwise combined) to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once. This provides a check of robustness of our performance.

## Pipeline
MLlib can represent a workflow as a Pipeline, which consists of a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order. A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage. For Transformer stages, the transform() method is called on the DataFrame. For Estimator stages, the fit() method is called.


	In the last, we fit the model and use the fitted model to make predictions on the test/holdout data.
	
	from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, IndexToString
	from pyspark.ml import Pipeline
	from pyspark.ml.classification import RandomForestClassifier
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	from pyspark.ml.regression import RandomForestRegressor
	from pyspark.ml.feature import VectorIndexer
	from pyspark.ml.evaluation import RegressionEvaluator
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator
	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


## Fit on whole dataset to include all labels in index.

	labelIndexer = StringIndexer(inputCol="spam_nospam", outputCol="indexedLabel").fit(df)



## exclude the label column and combine all the features to be used in a vector

	featuresCols = df.columns
	featuresCols.remove('spam_nospam')
	vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="Features")

## training the model
	
	rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="Features", numTrees=50)

## creating an evaluator for accuracy metric

	evaluator = MulticlassClassificationEvaluator(labelCol=rf.getLabelCol(), predictionCol=rf.getPredictionCol(), metricName="accuracy")


## 5- Fold Cross-Validation 

	paramGrid = ParamGridBuilder() \
	  .addGrid(rf.maxDepth, [2, 5])\
	  .build() # .addGrid(gbt.maxIter, [10, 100])

## Declare the CrossValidator, which runs model tuning for us.

	cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds= 5)


## Convert indexed labels back to original labels.
	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)


## Chain indexers, cross-validator and forest in a Pipeline

	pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, cv, labelConverter])


## Fit the model.  This also runs the indexers.

	model = pipeline.fit(train)

## Make predictions.
 
	predictions = model.transform(test)
	predictions.printSchema()
	

We can display the predictions and the features.
## Select example rows to display.
 
	predictions.select("predictedLabel", "spam_nospam", "Features").show(10)
+--------------+-----------+--------------------+
|predictedLabel|spam_nospam|            Features|
+--------------+-----------+--------------------+
|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)|           0.0|        0.0|(57,[54,55,56](#),[1...|
](#)+--------------+-----------+--------------------+
only showing top 10 rows

## Evaluating the Performance of the Model

We can evaluate the performance of our model on the following metric

1. Accuracy - Ratio of correctly predicted observations.
2. Precision - Precision looks at the ratio of correct positive observations. The formula is True Positives / (True Positives + False Positives).
3. Recall - Recall is also known as sensitivity or true positive rate. It’s the ratio of correctly predicted positive events. Recall is calculated as True Positives / (True Positives + False Negatives).
4. F1 measure - The F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. F1 is usually more useful than accuracy, especially if the class distribution is uneven.

	accuracy = evaluator.evaluate(predictions)
	print(accuracy)
0.926900584795
	predictions = predictions.withColumn('predictedLabel',predictions.predictedLabel.cast('double'))
	evaluator1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="predictedLabel", metricName="weightedPrecision")
	precision = evaluator1.evaluate(predictions)
	print precision
0.927923976608
	evaluator2 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="predictedLabel", metricName="weightedRecall")
	recall = evaluator2.evaluate(predictions)
	print recall
0.926900584795
	evaluator3 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="predictedLabel", metricName="f1")
	f1 = evaluator3.evaluate(predictions)
	print(f1)
0.926175392184