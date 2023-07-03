from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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


# frequency
retail_df_freq = retail_df_2.groupBy('CustomerID').agg(count('InvoiceDate').alias('frequency'))
# retail_df_freq.show(8)

retail_df_3 = retail_df_2.join(retail_df_freq, on='CustomerID', how='inner')
# data clearing - some rows have negative Quantity value - will be not included as errorneous
retail_df_3 = retail_df_3.where((retail_df_3['Quantity'] >= 0) & (retail_df_3['Quantity'] < 50000))
# retail_df_3.printSchema()

# monetary value - total amount spend by each customer
m_val_df = retail_df_3.withColumn('TotalAmount', col("Quantity")*col("UnitPrice"))
m_val_df = m_val_df.groupBy('CustomerID').agg(sum('TotalAmount').alias('monetary_value'))

# finally
final_df = m_val_df.join(retail_df_3, on='CustomerID', how='inner').select(['recency','frequency','monetary_value','CustomerID']).distinct()

# standartization and normalisation
assemble = VectorAssembler(inputCols=['recency', 'frequency', 'monetary_value'], outputCol='features')
assembled_data = assemble.transform(final_df)
# assembled_data.select('features').show(4, truncate=False)
scaler = StandardScaler(inputCol='features', outputCol='standardized')
data_scale = scaler.fit(assembled_data)
data_scale_output = data_scale.transform(assembled_data)
data_scale_output.select('standardized').show(4,truncate=False)

# looking for optimal cluster number with elbow method
""" Utility function to evaluate optimal number of clusters by elbow method 
    Args: maximal number of clusters to check, within range(2,max_num)
    Returns: y-data for ploptimal number of clusters
    Plots: the elbow plot
"""
def elbow_cluster_number(max_num):
    cluster_num = range(2,max_num+1)
    errors = []
    for n in cluster_num :
        model = KMeans(featuresCol='standardized',k=n).fit(data_scale_output)
        errors.append(model.summary.trainingCost)    
    opt_idx = 1
    delta_new = errors[0]-errors[1]
    delta_prev = 0
    while opt_idx < len(errors)-1 and delta_new >= delta_prev:
        delta_prev = delta_new
        opt_idx += 1
        delta_new = errors[opt_idx-1] - errors[opt_idx]
    opt_num = cluster_num[opt_idx-1]
    return errors, opt_num

# errors, opt_num = elbow_cluster_number(9)
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('SSE')
# plt.plot(range(2,10), errors)
# plt.show()
# print(f'optimal cluster number {opt_num} ')
opt_num = 3

# making predictions
model = KMeans(featuresCol='standardized',k=opt_num).fit(data_scale_output)
predict_df = model.transform(data_scale_output)
count_df = predict_df.groupBy('prediction').count().sort(asc('prediction')).show()
avg_m_val_df = predict_df.groupBy('prediction').agg(mean('monetary_value').alias('mean_value')).sort(asc('prediction')).show()


# viewing and saving results
vis_df = predict_df.select('recency','frequency','monetary_value','prediction').toPandas()
# 3D-view
cluster_fig_3d = plt.figure(figsize=(12,10))
ax3d = cluster_fig_3d.add_subplot(projection='3d')
ax3d.scatter(vis_df.frequency, vis_df.recency, vis_df.monetary_value,
             c=vis_df.prediction, depthshade=False)
ax3d.set_ylabel('recency')
ax3d.set_xlabel('frequency')
ax3d.set_zlabel('monetary_value')
plt.show()
cluster_fig_3d.savefig(f'./results/online-retail/3d_clusters.svg',format='svg')
# paired 2D-views
fig, axs = plt.subplots(1, 3, squeeze=False, figsize=(15, 5))
sns.set_theme(style='darkgrid')
sns.set_context("paper")

