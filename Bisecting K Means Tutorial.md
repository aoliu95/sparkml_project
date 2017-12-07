# Bisecting K Means Tutorial

## Introduction
spark.ml is a new package introduced in Spark 1.2, which aims to provide a uniform set of high-level APIs that help users create and tune practical machine learning pipelines. This tutorial covers an unsupervised machine learning algorithm from the sparkML library: Bisecting K Means.  The tutorial helps demonstrate the power of Spark and AWS by segmenting customers contained in an airline dataset containing over three million rows.  

## Unsupervised Learning on Airline Dataset
Most airlines are facing tough times due to increased competition and customer demands for a better experience. Cost is no longer the sole deciding factor between  organizations. We will be using customer segmentation to help the airline better understand its customers and tailor its offering by analyzing approximately 2 million trips taken by 1.5  million passengers.

## Dataset
We will use airline ticket data in this tutorial.  The focus of the tutorial is on the machine learning algorithm and Spark MLlib, therefore data cleaning and preparation were conducted separately and are not included in the tutorial.

Our dataset is a csv file that consists of airline ticket reservation information.  Each record has a PNR locator id associated with it and can have multiple tickets, users and legs for each PNR number. 

**We have the following information:**

|Feature | Description |
|:-------|:------------|
| PNRLocatorID | PNR locator id can have multiple tickets, users and legs. |
| uid | User ID. |
| PaxName | Deidentified Passenger Name |
| ServiceStartDate | When the flight takes off. |
| BookingChannel | How the passenger,booked the flight. |
| BaseFareAmt | Normalized Fare information. |
| UflyRewardsNumber | The rewards number,that was provided when booked. |
| UflyMemberStatus | The Ufly member,status. It will be either Standard or Elite. |
| age_group | Binned age of the passenger at the time of the flight. |
| true_origin | Airport code for segment start. |
| true_destination | Airport code for segment end. |
| group_size | Normalized group size. |
| group | Binary: 1 = group, 0 = individual. |
| seasonality | binned flying seasons, see below for breakdown |
| days_pre_booked | normalized, number of days between booking date and date of first flight segment. |
| origin_msp | Binary, true_origin is MSP = 1, all others = 0. |

### Keys for Feature Bins
####  Age Brackets:
0-17 -\>  0
18-24 -\> 0.11
25-34 -\> 0.22
35-64 -\>  0.48
65+ -\> 1.0

####  Seasonality:
Feb-May -\> 0
Jun-Aug -\> 0.33
Sep-Oct -\> 0.67
Nov-Jan -\> 1.0

####  Booking Channel:
Other -\> 0
Outside Booking -\>0.2
Tour Operator Portal -\> 0.4
SY Vacation -\> 0.6
Reservations Booking -\> 0.8
SCA Website Booking -\> 1.0

## Load the data and create a DataFrame. 
Dataframes provide a user-friendly API that is uniform across ML algorithms and across multiple languages. 

	#load the data file
	sun_country = sqlContext.read.format('com.databricks.spark.csv') \
	.option("inferSchema",True).option("header",True).load("s3://s3-ankit/SunCountryNormalized-All.csv")

## Explore the data and describe it. 
In this case, we processed 3 million rows of data.  Note that in the initial reading of the data all features default to string variables.

	#how many rows of data are in the DataFrame? and describe the DataFrame.
	sun_country.count()
3073826
	sun_country.describe()
Result:
	DataFrame[summary: string, _c0: string, PNRLocatorID: string, uid: string, PaxName: string, BookingChannel: string, BaseFareAmt: string, UFlyRewardsNumber: string, UflyMemberStatus: string, age_group: string, true_origin: string, true_destination: string, group_size: string, group: string, seasonality: string, days_pre_booked: string, origin_msp: string]
There is an option to view and create a Python Pandas DataFrame.  Pandas will automatically infer the type of each column. Note the revised data types. Here, we are just calling toPandas() to show how the function would infer the schema.  

	Convert the DataFrame to Python Pandas DataFrame
	sun_country.toPandas()
	sun_country.limit(2)
Result:
	DataFrame[_c0: int, PNRLocatorID: string, uid: string, PaxName: string, ServiceStartDate: timestamp, BookingChannel: double, BaseFareAmt: double, UFlyRewardsNumber: int, UflyMemberStatus: double, age_group: double, true_origin: string, true_destination: string, group_size: double, group: int, seasonality: double, days_pre_booked: double, origin_msp: int]
In the background, the dataframe consists of Row objects:
	for line in sun_country.take(5):
	print line
Result:
	Row(_c0=1, PNRLocatorID=u'AAAACD', uid=u'53544F555444696420493F7C2067657420746869732072696768745445524553412041F41411', PaxName=u'STOUTE', ServiceStartDate=datetime.datetime(2013, 8, 25, 0, 0), BookingChannel=0.2, BaseFareAmt=0.0918161308516638, UFlyRewardsNumber=0, UflyMemberStatus=0.0, age_group=0.48, true_origin=u'MDW', true_destination=u'SEA', group_size=0.1, group=0, seasonality=0.33, days_pre_booked=0.183823529411765, origin_msp=0)
	Row(_c0=2, PNRLocatorID=u'AAAACD', uid=u'53544F555444696420493F7C2067657420746869732072696768745445524553412041F41411', PaxName=u'STOUTE', ServiceStartDate=datetime.datetime(2013, 8, 25, 0, 0), BookingChannel=0.2, BaseFareAmt=0.0918161308516638, UFlyRewardsNumber=0, UflyMemberStatus=0.0, age_group=0.48, true_origin=u'MDW', true_destination=u'SEA', group_size=0.1, group=0, seasonality=0.33, days_pre_booked=0.183823529411765, origin_msp=0)
	Row(_c0=3, PNRLocatorID=u'AAAACD', uid=u'53544F555444696420493F7C2067657420746869732072696768745445524553412041F41411', PaxName=u'STOUTE', ServiceStartDate=datetime.datetime(2013, 8, 22, 0, 0), BookingChannel=0.2, BaseFareAmt=0.0918161308516638, UFlyRewardsNumber=0, UflyMemberStatus=0.0, age_group=0.48, true_origin=u'MDW', true_destination=u'SEA', group_size=0.1, group=0, seasonality=0.33, days_pre_booked=0.180672268907563, origin_msp=0)
	Row(_c0=4, PNRLocatorID=u'AAAACD', uid=u'53544F555444696420493F7C2067657420746869732072696768745445524553412041F41411', PaxName=u'STOUTE', ServiceStartDate=datetime.datetime(2013, 8, 22, 0, 0), BookingChannel=0.2, BaseFareAmt=0.0918161308516638, UFlyRewardsNumber=0, UflyMemberStatus=0.0, age_group=0.48, true_origin=u'MDW', true_destination=u'SEA', group_size=0.1, group=0, seasonality=0.33, days_pre_booked=0.180672268907563, origin_msp=0)
	Row(_c0=5, PNRLocatorID=u'AAAAMU', uid=u'4252554747454D414E44696420493F7C2067657420746869732072696768745045544552M48233', PaxName=u'BRUGPE', ServiceStartDate=datetime.datetime(2014, 5, 29, 0, 0), BookingChannel=1.0, BaseFareAmt=0.0687309644670051, UFlyRewardsNumber=0, UflyMemberStatus=0.0, age_group=0.22, true_origin=u'MSP', true_destination=u'DCA', group_size=0.1, group=0, seasonality=0.0, days_pre_booked=0.201680672268908, origin_msp=1)
To view the schema of the dataframe, we can use printSchema(). This operation displays the schema as a visual tree.
	sun_country.printSchema()
Result:
	root
	 |-- _c0: integer (nullable = true)
	 |-- PNRLocatorID: string (nullable = true)
	 |-- uid: string (nullable = true)
	 |-- PaxName: string (nullable = true)
	 |-- ServiceStartDate: timestamp (nullable = true)
	 |-- BookingChannel: double (nullable = true)
	 |-- BaseFareAmt: double (nullable = true)
	 |-- UFlyRewardsNumber: integer (nullable = true)
	 |-- UflyMemberStatus: double (nullable = true)
	 |-- age_group: double (nullable = true)
	 |-- true_origin: string (nullable = true)
	 |-- true_destination: string (nullable = true)
	 |-- group_size: double (nullable = true)
	 |-- group: integer (nullable = true)
	 |-- seasonality: double (nullable = true)
	 |-- days_pre_booked: double (nullable = true)
	 |-- origin_msp: integer (nullable = true)
## Create a subset of the dataframe with the relevant features.
Feature Selection was conducted separately.  The features determined to be most important in customer segmentation were the Booking Channel, Base Fare Amount, Age Group, Seasonality, Days PreBooked and if the Origin of the flight was Minneapolis/St Paul (Central Hub) or not. 
	cols = ['BookingChannel','BaseFareAmt','age_group','seasonality','days_pre_booked','origin_msp']
	df = sun_country.select(cols)
	df.limit(5).toPandas()
	df.printSchema()
Result:
	root
	 |-- BookingChannel: double (nullable = true)
	 |-- BaseFareAmt: double (nullable = true)
	 |-- age_group: double (nullable = true)
	 |-- seasonality: double (nullable = true)
	 |-- days_pre_booked: double (nullable = true)
	 |-- origin_msp: integer (nullable = true)
To further prepare the data for use in the model, we will import the pyspark functions, create a temporary table to capture and recast the features. Then we will need to convert the feature columns. The model expects strings, and doubles for numeric data. To do this we use cast().
	from pyspark.sql.functions import *
	
	df.registerTempTable("datatable")
	
	df = sqlContext.sql("""
	SELECT cast(BookingChannel as string), cast(age_group as string), BaseFareAmt, cast(seasonality as string), days_pre_booked, cast(origin_msp as string)
	from datatable
	""")
	
	df.printSchema()
Result:
	root
	 |-- BookingChannel: string (nullable = true)
	 |-- age_group: string (nullable = true)
	 |-- BaseFareAmt: double (nullable = true)
	 |-- seasonality: string (nullable = true)
	 |-- days_pre_booked: double (nullable = true)
	 |-- origin_msp: string (nullable = true)
## Convert the features to MLlib specific data types. 
Vectors are used to store features that will be used in the unsupervised algorithm.  The features will be prepared via Indexers, Encoders and an Assembler.  Finally, they will be fed into a pipeline for assembly. 
## String Indexer
The StringIndexer is used for converting categorical values into category indices. 

## OneHotEncoder
The one-hot encoder maps the column of category indices to a column of binary vectors, with at most a single one-value per row that indicates the input category index. 

## VectorAssembler
The vector assembler is a transformer that combines a given list of columns into a single vector column. We use it to combine all the raw features which we plan to use in our model. It is useful for combining raw features and features generated by different feature transformers into a single feature vector. 

## Pipeline
Finally, we will also use a pipeline which consists of a series of stages.  Each stage is either an estimator or transformer that runs in the order specified. The input DataFrame is transformed as it passes through each stage. For Transformer stages, the transform() method is called on the DataFrame. For Estimator stages, the fit() method is called.  

	from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
	OneHotEncoder,StringIndexer)
	BookingChannel_indexer = StringIndexer(inputCol="BookingChannel", outputCol="BookingChannel_indexer")
	age_group_indexer = StringIndexer(inputCol="age_group", outputCol="age_group_indexer")
	seasonality_indexer = StringIndexer(inputCol="seasonality", outputCol="seasonality_indexer")
	origin_msp_indexer = StringIndexer(inputCol="origin_msp", outputCol="origin_msp_indexer")

	BookingChannelEncoder = OneHotEncoder(inputCol = "BookingChannel_indexer", outputCol ="BookingChannelEncoder")
	AgeGroupEncoder = OneHotEncoder(inputCol = "age_group_indexer", outputCol ="AgeGroupEncoder")
	SeasonalityEncoder = OneHotEncoder(inputCol = "seasonality_indexer", outputCol ="SeasonalityEncoder")
	OriginMspEncoder = OneHotEncoder(inputCol = "origin_msp_indexer", outputCol ="OriginMspEncoder")
	assembler = VectorAssembler(inputCols = ["BookingChannelEncoder","BaseFareAmt","AgeGroupEncoder","SeasonalityEncoder","days_pre_booked","OriginMspEncoder"], 
	outputCol = "features")
	from pyspark.ml import Pipeline
	pipeline = Pipeline(stages=[BookingChannel_indexer, BookingChannelEncoder, age_group_indexer, AgeGroupEncoder, seasonality_indexer, SeasonalityEncoder, origin_msp_indexer, OriginMspEncoder, assembler])
	model=pipeline.fit(df)
	indexed=model.transform(df)

`indexed.limit(5).toPandas()`

  BookingChannel age_group  BaseFareAmt seasonality  days_pre_booked  \\
0            0.2      0.48     0.091816        0.33         0.183824   
1            0.2      0.48     0.091816        0.33         0.183824   
2            0.2      0.48     0.091816        0.33         0.180672   
3            0.2      0.48     0.091816        0.33         0.180672   
4            1.0      0.22     0.068731         0.0         0.201681   

  origin_msp  BookingChannel_indexer      BookingChannelEncoder  \\
0          0                     0.0  (1.0, 0.0, 0.0, 0.0, 0.0)   
1          0                     0.0  (1.0, 0.0, 0.0, 0.0, 0.0)   
2          0                     0.0  (1.0, 0.0, 0.0, 0.0, 0.0)   
3          0                     0.0  (1.0, 0.0, 0.0, 0.0, 0.0)   
4          1                     1.0  (0.0, 1.0, 0.0, 0.0, 0.0)   

   age_group_indexer       AgeGroupEncoder  seasonality_indexer  \\
0                0.0  (1.0, 0.0, 0.0, 0.0)                  1.0   
1                0.0  (1.0, 0.0, 0.0, 0.0)                  1.0   
2                0.0  (1.0, 0.0, 0.0, 0.0)                  1.0   
3                0.0  (1.0, 0.0, 0.0, 0.0)                  1.0   
4                1.0  (0.0, 1.0, 0.0, 0.0)                  0.0   

  SeasonalityEncoder  origin_msp_indexer OriginMspEncoder  \\
0    (0.0, 1.0, 0.0)                 1.0            (0.0)   
1    (0.0, 1.0, 0.0)                 1.0            (0.0)   
2    (0.0, 1.0, 0.0)                 1.0            (0.0)   
3    (0.0, 1.0, 0.0)                 1.0            (0.0)   
4    (1.0, 0.0, 0.0)                 0.0            (1.0)   

features  
0  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0918161308517, 1.0...  
1  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0918161308517, 1.0...  
2  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0918161308517, 1.0...  
3  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0918161308517, 1.0...  
4  (0.0, 1.0, 0.0, 0.0, 0.0, 0.068730964467, 0.0,...  

## Applying Bisecting k-means

Bisecting K-means can often be much faster than regular K-means, but it will generally produce a different clustering.  Bisecting k-means is a divisive form of hierarchical clustering. 

The implementation in MLlib has the following parameters:

-  setK: the desired number of leaf clusters (default: 4). The actual number could be smaller if there are no divisible leaf clusters.
-  setMaxIterations: the max number of k-means iterations to split clusters (default: 20)
-  minDivisibleClusterSize: the minimum number of points (if \>= 1.0) or the minimum proportion of points (if \< 1.0) of a divisible cluster (default: 1)
-  seed: a random seed (default: hash value of the class name)
## Import the algorithm modules and run the model:
-  In this case, the best fit for the number of clusters matches the default number of four clusters.  
-  After setting the parameters, fit and transform the model.  
-  The clusters are designated in the named prediction column.  We are calling it 'prediction'.
	from pyspark.ml.clustering import BisectingKMeans
	from pyspark.ml.clustering import KMeans
	
	
	bkm = BisectingKMeans().setK(4).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
	kmeans_bisecting = bkm.fit(indexed)
	oo = kmeans_bisecting.transform(indexed)

	oo.select('BookingChannel', 'BaseFareAmt', 'age_group', 'seasonality', 
	'days_pre_booked','origin_msp','features', 'prediction').show(truncate = False)
Result:
	+--------------+------------------+---------+-----------+-----------------+----------+----------------------------------------------------------------------------+----------+
	|BookingChannel|BaseFareAmt       |age_group|seasonality|days_pre_booked  |origin_msp|features                                                                    |prediction|
	+--------------+------------------+---------+-----------+-----------------+----------+----------------------------------------------------------------------------+----------+
	|0.2           |0.0918161308516638|0.48     |0.33       |0.183823529411765|0         |(15,[0,5,6,11,13],[1.0,0.0918161308516638,1.0,1.0,0.183823529411765])       |2         |
	|0.2           |0.0918161308516638|0.48     |0.33       |0.183823529411765|0         |(15,[0,5,6,11,13],[1.0,0.0918161308516638,1.0,1.0,0.183823529411765])       |2         |
	|0.2           |0.0918161308516638|0.48     |0.33       |0.180672268907563|0         |(15,[0,5,6,11,13],[1.0,0.0918161308516638,1.0,1.0,0.180672268907563])       |2         |
	|0.2           |0.0918161308516638|0.48     |0.33       |0.180672268907563|0         |(15,[0,5,6,11,13],[1.0,0.0918161308516638,1.0,1.0,0.180672268907563])       |2         |
	|1.0           |0.0687309644670051|0.22     |0.0        |0.201680672268908|1         |(15,[1,5,7,10,13,14],[1.0,0.0687309644670051,1.0,1.0,0.201680672268908,1.0])|0         |
	|1.0           |0.0687309644670051|0.22     |0.0        |0.20063025210084 |1         |(15,[1,5,7,10,13,14],[1.0,0.0687309644670051,1.0,1.0,0.20063025210084,1.0]) |0         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.20063025210084 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.20063025210084,1.0])          |1         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.19327731092437 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.19327731092437,1.0])          |1         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.19327731092437 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.19327731092437,1.0])          |1         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.20063025210084 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.20063025210084,1.0])          |1         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.20063025210084 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.20063025210084,1.0])          |1         |
	|1.0           |0.10545967287084  |0.48     |0.67       |0.19327731092437 |1         |(15,[1,5,6,13,14],[1.0,0.10545967287084,1.0,0.19327731092437,1.0])          |1         |
	|1.0           |0.124608009024253 |0.22     |0.33       |0.16281512605042 |1         |(15,[1,5,7,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.16281512605042,1.0])  |0         |
	|1.0           |0.124608009024253 |0.22     |0.33       |0.168067226890756|1         |(15,[1,5,7,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.168067226890756,1.0]) |0         |
	|1.0           |0.124608009024253 |0.0      |0.33       |0.168067226890756|1         |(15,[1,5,8,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.168067226890756,1.0]) |0         |
	|1.0           |0.124608009024253 |0.0      |0.33       |0.16281512605042 |1         |(15,[1,5,8,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.16281512605042,1.0])  |0         |
	|1.0           |0.124608009024253 |0.48     |0.33       |0.16281512605042 |1         |(15,[1,5,6,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.16281512605042,1.0])  |1         |
	|1.0           |0.124608009024253 |0.48     |0.33       |0.168067226890756|1         |(15,[1,5,6,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.168067226890756,1.0]) |1         |
	|1.0           |0.124608009024253 |0.48     |0.33       |0.168067226890756|1         |(15,[1,5,6,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.168067226890756,1.0]) |1         |
	|1.0           |0.124608009024253 |0.48     |0.33       |0.16281512605042 |1         |(15,[1,5,6,11,13,14],[1.0,0.124608009024253,1.0,1.0,0.16281512605042,1.0])  |1         |
	+--------------+------------------+---------+-----------+-----------------+----------+----------------------------------------------------------------------------+----------+
## Explore the Result
Cluster Centers can be verified with the **clusterCenters()** command.
	centers = kmeans_bisecting.clusterCenters()
	print("Cluster Centers: ")
	for center in centers:
	print(center)

**computeCost()** will return the Bisecting K-means cost.  It is the sum of squared distances of points to their nearest center for this model on the given data. If provided with an RDD of points returns the sum.

	kmeans_bisecting.computeCost(indexed)
Result:
4112461.7713863174
	bkm2 = BisectingKMeans().setK(3).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction") 
	kmeans_bisecting2 = bkm2.fit(indexed)
	oo2 = kmeans_bisecting2.transform(indexed)
	kmeans_bisecting2.computeCost(indexed)