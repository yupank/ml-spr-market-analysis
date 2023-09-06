from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

from ml_utils import logreg_analyze

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler

import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

file_path = './data/bank-marketing.txt'
# file_path = './data/bank-full.txt'
bank_df = spark.read.option("header", True).option("inferSchema", True).option("sep",';').csv(file_path)
""" refactoring months to more sensible format and indexing"""
month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_nums = [str(i+1) for i in range(12)]
bank_df = bank_df.replace(month_names,month_nums)
# si_mon = StringIndexer(inputCol='month', outputCol='month_Idx', stringOrderType='alphabetAsc')
# bank_df = si_mon.fit(bank_df).transform(bank_df)
# bank_df.show(5)

""" preparing features for ML"""
# selecting only the client data for 1st part of  ML analysis 
# Hypothesis_0 - marketing campain does not affect the subscription
client_info_cols = bank_df.columns[:8]
client_info_cols.append('decision')
ml_client_df = bank_df.select(client_info_cols)
# converting categorical variables into numerical form
cat_cols_in = client_info_cols[1:5] + ['housing', 'loan', 'decision']
cat_cols_idx = [col+'_Idx' for col in cat_cols_in]
cat_cols_vec = [col+'_Vec' for col in cat_cols_in]
for col_in in cat_cols_in:
    si_cat = StringIndexer(inputCol=col_in,outputCol=col_in +'_Idx')
    ml_client_df = si_cat.fit(ml_client_df).transform(ml_client_df)
# One-Hot encoding
encoder = OneHotEncoder(inputCols=cat_cols_idx, outputCols=cat_cols_vec)
ml_client_df = encoder.fit(ml_client_df).transform(ml_client_df)
# assembling the features vector
vec_cols = ['age', 'balance']+cat_cols_vec

ml_assembler = VectorAssembler(inputCols=vec_cols, outputCol="features")
ml_client_df = ml_assembler.transform(ml_client_df)
ml_client_df = ml_client_df.select(client_info_cols + ['features', 'decision_idx'])

ml_client_df.show(10,False)

model_1_df = ml_client_df.select('features', 'decision_idx')
train_df,test_df=model_1_df.randomSplit([0.75,0.25])
# results, cm, accuracy, recall = logreg_analyze(train_df, test_df)
# print(results.sample(fraction=0.05).show(10))
# print(f'accuracy: {accuracy}, recall: {recall}')

# now including the campain data, Hypothesis_1 - campain have an effect
ml_campaign_df = bank_df.drop('pdays', 'previous', 'poutcome')
#updating the categorical columns and feature vectors


cat_cols_in.append('month')
cat_cols_idx = [col+'_Idx' for col in cat_cols_in]
cat_cols_vec = [col+'_Vec' for col in cat_cols_in]
for col_in in cat_cols_in:
    si_cat = StringIndexer(inputCol=col_in,outputCol=col_in +'_Idx',stringOrderType='alphabetAsc')
    ml_campaign_df = si_cat.fit(ml_campaign_df).transform(ml_campaign_df)
# One-Hot encoding
encoder = OneHotEncoder(inputCols=cat_cols_idx, outputCols=cat_cols_vec)
ml_campaign_df = encoder.fit(ml_campaign_df).transform(ml_campaign_df)
# assembling new features vector
vec_cols = ['age', 'balance', 'duration', 'campaign']+cat_cols_vec
ml_assembler = VectorAssembler(inputCols=vec_cols, outputCol="features")
ml_campaign_df = ml_assembler.transform(ml_campaign_df)

campaign_info_cols = client_info_cols + ['month', 'duration', 'campaign']
ml_campaign_df = ml_campaign_df.select(client_info_cols + ['features', 'decision_idx'])
ml_campaign_df.show(10, False)

model_2_df = ml_campaign_df.select('features', 'decision_idx')
train_df,test_df=model_2_df.randomSplit([0.75,0.25])
results, cm, accuracy, recall = logreg_analyze(train_df, test_df)
print(results.sample(fraction=0.05).show(10))
print(f'accuracy: {accuracy}, recall: {recall}')