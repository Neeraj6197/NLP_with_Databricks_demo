# Databricks notebook source
#importing the delta table:
data = spark.table('spam_text')
data.display()

# COMMAND ----------

#checking the size of dataset:
data.count()

# COMMAND ----------

#importing required libraries for preprocessing the data:
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# COMMAND ----------

data.select('Message').collect()[0][0]

# COMMAND ----------

#defining a UDF function to preprocess the data:
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType,StringType

ps = PorterStemmer()

@udf()
def preprocessor(message):
    #removing the special characters:
    text = re.sub('[^a-zA-Z0-9]',' ',message)
    #lowering the case:
    text = text.lower()
    #splitting the text:
    text = text.split()
    #stemming and removing the stopwords:
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    #rejoining the words to sentences:
    text = ' '.join(text)
    return text

# COMMAND ----------

#applying the udf:
from pyspark.sql.functions import col
preprocessed_data = data.withColumn('Message',preprocessor(data['Message']))
preprocessed_data = preprocessed_data.withColumn('Message',col('Message').cast('array<string>'))
preprocessed_data.display()

# COMMAND ----------

#applying count vectorizer(bag of words):
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="Message", outputCol="features")
model = cv.fit(preprocessed_data)
result_df = model.transform(preprocessed_data)
result_df.display()

# COMMAND ----------

data.dtypes

# COMMAND ----------


