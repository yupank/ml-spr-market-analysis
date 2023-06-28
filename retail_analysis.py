from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()
# spark = SparkSession.builder.appName("pyspark ML app").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()

retail_df = spark.read.csv('./data/online_retail.csv', header=True)

""" Exploratory Analysis"""
# retail_df.show(5,0)
# number of unique customers
print(f"unique customers: {retail_df.select('CustomerId').distinct().count()}")

# country with most purchase made
# retail_df.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).orderBy(desc('country_count')).show(10)

""" Clustering Analysis"""
# data preprocessing 
# recency
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
retail_df = retail_df.withColumn('date', to_timestamp("InvoiceDate",'dd/MM/yy HH:mm'))
# retail_df.select(min("date")).show()
# retail_df.select(max("date")).show()

retail_df = retail_df.withColumn("from_date", lit("10/1/12 08:26"))
retail_df = retail_df.withColumn('from_date',to_timestamp("from_date", 'yy/MM/dd HH:mm'))
retail_df_2 = retail_df.withColumn('from_date',to_timestamp(col('from_date'))).withColumn('recency',col("date").cast("long") - col('from_date').cast("long"))

retail_df_2 = retail_df_2.join(retail_df_2.groupBy('CustomerID').agg(max('recency').alias('recency')),on='recency',how='leftsemi')
# retail_df_2.show(8)

# frequency
retail_df_freq = retail_df_2.groupBy('CustomerID').agg(count('InvoiceDate').alias('frequency'))
retail_df_freq.show(8)

retail_df_3 = retail_df_2.join(retail_df_freq, on='CustomerID', how='inner')
retail_df_3.printSchema()

# monetary value
monetary_val = retail_df_3.withColumn('TotalAmount', col("Quantity")*col("UnitPrice"))