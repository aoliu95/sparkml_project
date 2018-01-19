---
Authors: Abhijit C Patil, Abhilasha Pandey, Ankit Agarwal, Ao Liu, Suzanne Kaminski,
Date: December 06, 2017
Title: Helping Sun Country understand its customers using clustering and LDA on PySpark AND a tutorial to demonstrate Random Forest using PySpark

---

# Introduction

Data analytics and machine learning are one of the most important applications of distributed computing. **spark.ml** is a new package introduced in Spark 1.2+, which aims to provide a uniform set of high-level APIs that help users create and tune practical machine learning pipelines.We wanted to learn and explore SparkML in detail.  To better demonstrate the vast capabilities in spark.ml, three tutorials were created that cover both supervised and unsupervised techniques.  Algorithms included are Random Forest, Bisecting k-means and LDA. **Our project contained topics within both Categories A and B**.

To demonstrate the power of Spark and AWS together and show how they leverage big data sets, the project completed customer segmentation of a Sun Country Airlines dataset. In concert with analyzing the customer segments and to provide a more complete business solution to Sun Country, topic modeling and sentiment analysis were conducted after scraping airline reviews from the web. 

# Analytics with Big Data (Goal B)

### Background
Most airlines are facing tough times due to increased competition and customer demands for a better experience. Cost is no longer the sole deciding factor between  organizations. We want to help Sun Country to improve the experience its customers have on the digital platforms and increase adoption of its UFLY membership program. We will be using customer segmentation and topic modelling to help Sun Country better understand its customers and tailor its offering by analyzing approximately 2 million trips taken by 1.5  million passengers. We want to use the data to segment customers of Sun Country Airlines into general categories of people with similar flying patterns. 
Passenger airlines is a highly competitive industry. For a relatively smaller airlines like Sun Country optimizing each and every task is the way to profitability. Having served customers for over 20 years Sun Country has a rich customer information. The goal of this analysis is to identify meaningful and actionable group of customers. It is also important to know what are the characteristics that set these groups apart. The marketing managers at Sun Country would like to understand the customer patterns so they can market them more efficiently. Following table presents more details about the task at hand.
2.1.5 Data overview 
The Sun Country dataset includes booking information along with demographic and some derived metrics. Following are some of the prominent features in the data:

### Data overview 
The Sun Country dataset includes booking information along with demographic and some derived metrics. Following are some of the prominent features in the data:

![Featues](https://github.umn.edu/patil074/Maroon02/blob/master/Images/kmeansFeatures.PNG)


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

### Methodology 

1. **Data Preparation** - Analyzing the Sun Country dataset started with preprocessing and cleaning the data.  Faulty values were examined and removed or replaced.  Faulty age values were replaced with the median age.  NA values in the Ufly membership column were replaced with a binary code of 0.  Duplicate PNR values were removed.
2. **Data Transformation** - Certain features were transformed by binning and then all features were normalized.  Clustering is particularly sensitive to differences in scale..  Age and Booking Date were binned and group size was transformed to a binary feature of group or single traveller.  A new binary feature was created to describe whether a trip started in Minneapolis (MSP) (headquarters and main hub) or outside of MSP.  
3. **Modeling** - See the Appendix for a detailed accounting of the modelling process.  After preparation, the data was explored and the relevant features are determined.  In this case, the Booking Channel, Base Fare, Age Group, Seasonality, Days Pre Booked and the Origin City were selected. The features are stored in vectors via Indexers, Encoders and an Assembler, and finally fed into a pipeline.  After exploration it was determined that the best number of clusters is the default of four.

A step by step guideline to implementing bisecting k-means using pyspark can be found [here](https://github.umn.edu/patil074/Maroon02/blob/master/Tutorials/Bisecting%20K-Means%20Tutorial.md)

### Findings  

With the help of K- Means Bisecting Algorithm, we were able to identify four distinct clusters. 
Out of the four clusters, two of them were fuzzy and not very distinct with the exception of one cluster that contained customers originating outside of Minneapolis.  This cluster also had a low rewards program membership. This would be a prime opportunity for the marketing team. The other two clusters, were quite distinct and actionable. Another of the clusters was "Vacationing Grandparents", who booked their tickets during February- May at very high prices. This cluster used third party booking channels for their booking. The other distinct cluster was "Young Professionals" who belong to the age group of 24-35. These professionals tend to spend less money on their tickets and make their booking using SCA website

![Features](https://github.umn.edu/patil074/Maroon02/blob/master/Images/KmeansFindings.PNG)

Through the 'vacationing grandparent' cluster we came to know that one Sun country's major customer segment does not use their website. This might indicate that SCA website is not user friendly and therefore, Sun Country can try improving the experience for such customers.


## Latent Dirichlet Allocation

### Background
Customer feedback is very important for any company operating in the service industry. While Sun Country is adding more customers, flights, and destinations under its banner maintaining quality service is of prime importance. The best way to address customers' issues is by listening to them. It is physically impossible and not financially viable to listen and note what each customer is saying nor does each and every customer leaves a feedback. However, there is a sizable population of customers who review/comment on the service on many third-party website. Sun Country would like to understand what people are saying in these feedbacks. We used LDA on Spark by scraping reviews for Sun Country airlines from tripadvisor.com. We these reviews as a collection of documents and run LDA on it.

###  What it is Latent Dirichlet Allocation?
Topic modeling is an unsupervised machine learning algorithm to find abstract topics that occur in a collection of documents. 

Latent Dirichlet allocation (LDA) is a statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. Latent Dirichlet allocation is one of the most common algorithms for topic modeling.  LDA is used to identify major topics from a collection of texts. For example, using LDA you can identify the major gist of a new article. This article can be scaled up and down as required.

LDA treats each document as a mixture of topics, and each topic as a mixture of words. This allows documents to overlap each other in terms of content, rather than being separated into discrete groups, in a way that mirrors typical use of natural language.

### Data overview 
The data here is a collection of reviews left by customers of Sun Country airlines on Tripadvisor. The data collected from the website is stored in a csv and then imported in Spark. The collection of reviews stored as RDD. 
Data Sample:

![Features](https://github.umn.edu/patil074/Maroon02/blob/master/Images/LDADatasample.PNG)

### Goal of analysis
The goal of the analysis is to identify major concerns the customers are expressing on the review websites. The airlines can use these as feedback to improve its customer service and thus improving customer satisfaction. 


### Methodology 
1. Before running LDA we need to follow a few data process steps. 
2. The first one is normalising the data. Normalising is a process of converting text into single canonical form. This makes it easier to interpret the text.
3. You are also needed to removed all the punctuations like (.), (,), (;) and others. This is done since there is no thought captured in these characters.
4. Removing stop words is also important. This includes removing all the articles and prepositions
5. Lastly we need to stem words. Stemming is the process of converting the word into its base form.
6. We now have a bag of words which we run LDA on. 
7. The hyper parameters for LDA include the number of topics and the words per topic

![Features](https://github.umn.edu/patil074/Maroon02/blob/master/Images/LDAProcess.PNG)

A step by step guideline to implementing LDA using pyspark can be found [here](https://github.umn.edu/patil074/Maroon02/blob/master/Tutorials/LDA%20Tutorial.md)

### Findings
From the LDA analysis on reviews we find that there are three major topics being discussed by the customers:

| Crew Appreciation | Vacation | Timely performance | 
|:---------|:---------|:---------|
| Seemed | 	Mexico | 	Appreciate | 
| Best | 	Vacation | 	Early | 
| Been | 	Took | 	Sure | 
| Crew | 	Also | 	Ticket | 
| Think | 	Lots | 	Plane | 

The above groups are defined by the user depending on the topics that are churned out of the analysis.



## A tutorial on Random Forest
### What it is Random Forest?
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.

### What is Random Forest used for?
Random forest is a commonly used supervised machine learning technique. It is used to classify unknown data into labels of events and non-events. Like on our case we will use a wide array of available features to train the algorithm to find spam emails.

### How does Random Forest work?
Random Forests are similar to a famous Ensemble technique called Bagging but have a different tweak in it. In Random Forests the idea is to decorrelate the several trees which are generated on the different bootstrapped samples from training Data.And then we simply reduce the Variance in the Trees by averaging them.
Averaging the Trees helps us to reduce the variance and also improve the Performance of Decision Trees on Test Set and eventually avoid Overfitting.

The idea is to build lots of Trees in such a way to make the Correlation between the Trees smaller.

### Data overview
To demonstrate the working of Random Forest classification technique we will use an open source data from UCI. You can download the data from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase).

There are 4,601 observations in the data. Of the total 4,601 observations 1,813 (39.4%) are spam. There are 58 attributes/features available for the analysis. 57 are continuous class variables whereas the last one is a nominal class variable indicating whether the email was spam.

Following are the details about attributes available for the analysis:
1. 48 continuous real [0,100] attributes of type word_freq_WORD = percentage of words in the e-mail that match WORD,i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail.  A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string. 
2. 6 continuous real [0,100] attributes of type char_freq_CHAR= percentage of characters in the e-mail that match CHAR,i.e. 100 * (number of CHAR occurrences) / total characters in email.
3. 1 continuous real [1,...] attribute of type capital_run_length_average= average length of uninterrupted sequences of capital letters
4. 1 continuous integer [1,...] attribute of type capital_run_length_longest= length of longest uninterrupted sequence of capital letters
5. 1 continuous integer [1,...] attribute of type capital_run_length_total = sum of length of uninterrupted sequences of capital letters = total number of capital letters in the e-mail
6. 1 nominal {0,1} class attribute of type spam = denotes whether the email was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. 

For more information, see file 'spambase.DOCUMENTATION' at the [UCI Machine Learning Repository](http://www.ics.uci.edu/~mlearn/MLRepository.html)

### Goal of analysis
The goal of this analysis is to develop a tutorial to help people understand how to implement Random Forest using pyspark. We have executed the tutorial by creating a custom spam filter using machine learning algorithm using the features calculated from the text in the mail.

### Overview of the procedure/findings, etc
We followed the standard data pre processing techniques to clean and set up the data for machine learning.

Data Pre-processing 

![Features](https://github.umn.edu/patil074/Maroon02/blob/master/Images/RandomForestDataPrep.PNG)

### Methodology 
Once this is ready you can pass the data to the algorithm. Here we use 5 fold cross validation technique. 
A complete step by step guide can be found [here](https://github.umn.edu/patil074/Maroon02/blob/master/Tutorials/Random%20Forest%20tutorial.md)


### Findings
After running the model, we find that the accuracy is 92%. From one of the iterations we see the confusion matrix as below.

| Measure | Value |
|:--------|------:|
Accuracy | 92.7% |
Recall | 86.0% |
Precision | 94.6% |

# Concluding remarks 
SparkML is a strong platform for processing and analyzing large datasets.  Spark was able to process over three million rows of data in a matter of minutes.  This is in direct contrast to other platforms like R Studio where we struggled to process only 10,000 rows of data.  Spark.ml allowed us to provide Sun Country with actionable and useful business case suggestions that will help them target their marketing and increase customer awareness and membership in their Ufly program.  In addition to the customer insights and ability to leverage big data tools for the benefit of the client, we have also provided three tutorials that will be available for those that wanting to learn spark mllib and take advantage of the power and efficiency that it can provide.  
In summary, we found two distinct clusters which can be used to  create customized marketing strategy to improve customer experience.  Also Sun Country could modify its website to make it more user friendly to ensure the older customers also book through the website. Marketing to origin cities outside the Twin Cities will provide an opportunity to grow awareness and brand loyalty through the Ufly program.  Also through using LDA we observed that the sentiment related to Sun Country is mostly positive and it would be a good approach for them to periodically run it to see how the customers feel about the brand. 
Big Data related Spark could bring huge potentials to the airlines industry. Marketing decisions that used to take weeks to decide now would only take seconds to generate. Meanwhile, with real-time analysis and customer clustering, airlines would benefit from personalized flight offer, better customer service and higher operational efficiency.

We learned how Big Data can be leveraged to solve analytics problem. Using Big Data, Spark and ML Algorithms, we were able to read, process and cluster 2.5 millions rows, under 25 minutes.

We also explored Qubole, which is a cloud agnostic third party software which helped our team to collaborate real-time on the same notebook and reduced the complication involved in sharing notebooks.

The major challenge our team faced in this project was related to setting up the environment. We realized that the documentation for most services is still very raw and rare as most services are still in nascent phases. Thus we have created a setup guide for the technologies we explored and can be found here:
1. [Qubole](https://github.umn.edu/patil074/Maroon02/blob/master/Handouts/Maroon%2002%20Qubole%20AWS%20Quick%20setup%20Jupyter%20Notebook.pdf )
2. [Google Big Cloud](https://github.umn.edu/patil074/Maroon02/blob/master/Handouts/Maroon%2002%20Google%20Cloud%20Quick%20setup%20Jupyter%20Notebook.pdf )

# Bibliography

1. https://archive.ics.uci.edu/ml/machine-learning-databases/spambase
2. https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/
3. https://www.r-bloggers.com/random-forests-in-r/
4. https://spark.apache.org/docs/latest/ml-guide.html
