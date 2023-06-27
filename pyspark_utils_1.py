from pyspark.sql import SparkSession
from pyspark.sql import Row
from datetime import datetime, date
import pandas as pd
from pyspark.sql.functions import pandas_udf

spark = SparkSession.builder.getOrCreate()

""" creating spark DataFrames from different sources """
df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')
# df.show()
""" creating spark DataFrame from pandas dataframe """
pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df = spark.createDataFrame(pandas_df)
df.show()
df.printSchema()

""" applyinng a function """
@pandas_udf('long')
def pd_plus_one(series: pd.Series) ->pd.Series:
    return series + 1
# df.select(pd_plus_one(df.a)).show()


""" reading the data from csv directly """
retail_df = spark.read.csv('./data/online_retail.csv', header=True)
# print(retail_df.columns)
# retail_df.show(5)
# retail_df.select("Description","Quantity","UnitPrice").describe().show(5)
retail_df.printSchema()

""" running  SQL queries """
retail_df.createOrReplaceTempView("retail_table")
spark.sql("SELECT Country, count(InvoiceNo) From retail_table GROUP BY Country ORDER BY Country LIMIT 10").show()