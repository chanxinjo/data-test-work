from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types

spark = SparkSession.builder.appName('test_data_enrichment').getOrCreate()

loyalty = spark.read.option('header', 'true').csv('test_dataset/loyalty.csv')
transactions = spark.read.option('header', 'true').csv('test_dataset/transactions.csv')

# get scraped dataset
city_state = spark.read.option('header', 'true').csv('enrich_dataset/us_census_city')
city_state = city_state.select('NAME', 'STNAME')

# distinct city
city_distinct = transactions.select('city').distinct()

# join
join_df = city_distinct.join(city_state)

def _data_match(partial_str, full_str):
    partial = ''.join([x.upper() for x in partial_str if x.isalpha()])
    full = ''.join([x.upper() for x in full_str if x.isalpha()])

    if partial in full:
        return True
    else:
        return False
data_match = F.udf(_data_match, types.BooleanType())

join_df = join_df.withColumn('match', data_match(F.col('city'), F.col('NAME')))

city_state_data = join_df.where(F.col('match') == True)
city_state_data = city_state_data.groupBy('city').agg(
    F.collect_set(F.col('STNAME')).alias('states_list')
)
city_state_data = city_state_data.withColumn('possible_states', F.concat_ws(',', F.col('states_list')))
city_state_data = city_state_data.select(
    F.col('city').alias('city_al'),
    F.col('possible_states')
)

enriched_transactions = transactions.join(city_state_data, transactions['city'] == city_state_data['city_al'], how='left').drop('city_al')

# user would be able to populate state data from the customer base
# for example
enriched_transactions.groupBy('possible_states').count()
