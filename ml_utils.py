from pyspark.sql.types import *
from pyspark.sql.functions import *


from pyspark.ml.classification import LogisticRegression

""" basic description of client data and 
    basic analysis - is there any effects of age, education, etc..."""
def analyze_df_basics(bank_df):
    # basic description
    bank_df.show(8)
    bank_df.printSchema()
    #splitting in two groups by the main outcome
    deposit_y_df = bank_df.where(bank_df['decision'] == 'yes')
    deposit_n_df = bank_df.where(bank_df['decision'] == 'no')
    num_clients_y = deposit_y_df.select('age').count()
    num_clients_n = deposit_n_df.select('age').count()
    print(f"subscribed yes/no: {num_clients_y} / {num_clients_n}")
    deposit_y_df.select('age').describe().show()

    #differerence in the level of education
    print("subscribed:\n")
    edu_y_agg = deposit_y_df.groupby('education').agg((count('education')/num_clients_y).alias('fraction'))
    edu_y_agg.show()
    print("not subscribed:\n")
    edu_n_agg = deposit_n_df.groupby('education').agg((count('education')/num_clients_n).alias('fraction'))
    edu_n_agg.show()


""" trains and test logistic regression model for specified features and labels columns """
def logreg_analyze(train_df, test_df, feature_cols='features',labels_col='decision_idx'):
    # train_df.sample(fraction=0.01).show(10)
    # test_df.sample(fraction=0.05).show(10)

    log_reg = LogisticRegression(labelCol=labels_col).fit(train_df)
    train_results = log_reg.evaluate(train_df).predictions
    test_results = log_reg.evaluate(test_df).predictions
    # some insights into probabilities
    # test_results.filter( 
    #     (test_results[labels_col]==1)& (test_results.prediction==1) 
    #     ).select(labels_col, 'prediction', 'probability').show(10, False)
    # confusion matrix
    print("confusion matrix: ")
    cm_df = test_results.groupBy(labels_col, "prediction").count().show()
    # just in case, when confusion matrix seems too good to be true
    true_pos = test_results.filter( (test_results[labels_col]==1)& (test_results.prediction==1)).count()
    true_neg = test_results.filter( (test_results[labels_col]==0)& (test_results.prediction==0)).count()
    false_pos = test_results.filter( (test_results[labels_col]==0)& (test_results.prediction==1)).count()
    false_neg = test_results.filter( (test_results[labels_col]==1)& (test_results.prediction==0)).count()
    # print(f'TP:{true_pos}, TN {true_neg}, FP:{false_pos}, FN:{false_neg}')
    accuracy = float((true_pos + true_neg)/(test_results.count()))
    recall = float( true_pos/(true_pos + false_neg) )
    
    return test_results, cm_df, accuracy, recall

# @pandas_udf('long')
# def days_past(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
#     month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']
#     for day, month in iterator:
#         idx = month_names.index(month)
#         if idx > 10 :
#             idx = -1
#         days = 31 - day + (10 - idx)*30
#         yield days