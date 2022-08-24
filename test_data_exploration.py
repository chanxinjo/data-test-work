from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window, types

spark = SparkSession.builder.appName('test_data_exploration').getOrCreate()

loyalty = spark.read.option('header', 'true').csv('test_dataset/loyalty.csv')
transactions = spark.read.option('header', 'true').csv('test_dataset/transactions.csv')

######################################################################################
# check if there are any duplicates in column id
duplicate_loyalty = loyalty.groupBy('id').count().where(F.col('count') > 1)
print(f'Duplicate ID in loyalty dataset: {duplicate_loyalty.count()}')

duplicate_transactions = transactions.groupBy('id').count().where(F.col('count') > 1)
print(f'Duplicate ID in transactions dataset: {duplicate_transactions.count()}')

# Result:
# Duplicate ID in loyalty dataset: 0
# Duplicate ID in transactions dataset: 0

# Justification:
# No duplicate IDs, assuming only one-to-one relationship
######################################################################################

######################################################################################
# try join using column id in both datasets
loyalty_columns = [F.col(col).alias(f'loyalty_{col}') for col in loyalty.columns]
loyalty_df = loyalty.select(*loyalty_columns)

transactions_columns = [F.col(col).alias(f'transactions_{col}') for col in transactions.columns]
transactions_df = transactions.select(*transactions_columns)

join_df = loyalty_df.join(transactions_df, loyalty_df['loyalty_id'] == transactions_df['transactions_id'])
print(f'Inner join datasets using column ID results in count: {join_df.count()}')

# Result:
# Inner join datasets using column ID results in count: 10000

# Justification:
# All records are referencable to each other using column id
######################################################################################

######################################################################################
# create UDF for data cross checking
def _compare_string(processed_str, clean_str):
    if processed_str == clean_str:
        return []
    else:
        noise_list = []
        noise_str = ''
        clean_index = 0
        for char in processed_str:
            if clean_index >= len(clean_str):
                noise_str += char
            elif char == clean_str[clean_index]:
                clean_index += 1
                if noise_str != '':
                    noise_list.append(noise_str)
                    noise_str = ''
            else:
                noise_str += char

        if noise_str != '':
            noise_list.append(noise_str)

        if clean_index != len(clean_str):
            return None

        return noise_list
compare_string = F.udf(_compare_string, types.ArrayType(types.StringType()))

# compare name
join_df = join_df.withColumn('compare_name', F.when(F.col('loyalty_name') == F.col('transactions_name'), F.lit(True)).otherwise(F.lit(False)))
join_df = join_df.withColumn('name_diff', compare_string(F.col('loyalty_name'), F.col('transactions_name')))

# compare city
join_df = join_df.withColumn('compare_city', F.when(F.col('loyalty_city') == F.col('transactions_city'), F.lit(True)).otherwise(F.lit(False)))
join_df = join_df.withColumn('city_diff', compare_string(F.col('loyalty_city'), F.col('transactions_city')))

# compare phone number
join_df = join_df.withColumn('compare_phone-number', F.when(F.col('loyalty_phone-number') == F.col('transactions_phone-number'), F.lit(True)).otherwise(F.lit(False)))
join_df = join_df.withColumn('phone-number_diff', compare_string(F.col('loyalty_phone-number'), F.col('transactions_phone-number')))

# compare email
join_df = join_df.withColumn('compare_email', F.when(F.col('loyalty_email') == F.col('transactions_email'), F.lit(True)).otherwise(F.lit(False)))
join_df = join_df.withColumn('email_diff', compare_string(F.col('loyalty_email'), F.col('transactions_email')))
######################################################################################

######################################################################################
# validation
# ensure that no additional character in transactions dataset when compared to loyalty dataset
name_not_tally = join_df.where(F.col('name_diff').isNull())
print(f'Data not tally in column name: {name_not_tally.count()}')

city_not_tally = join_df.where(F.col('city_diff').isNull())
print(f'Data not tally in column city: {city_not_tally.count()}')

phone_not_tally = join_df.where(F.col('phone-number_diff').isNull())
print(f'Data not tally in column phone-number: {phone_not_tally.count()}')

email_not_tally = join_df.where(F.col('email_diff').isNull())
print(f'Data not tally in column email: {email_not_tally.count()}')

# Result:
# Data not tally in column name: 0
# Data not tally in column city: 0
# Data not tally in column phone-number: 0
# Data not tally in column email: 0

# Justification:
# All data tally (UDF will return NULL if there is additional character in transactions dataset not captured in loyalty dataset)
######################################################################################

######################################################################################
# general report
clean_data = join_df.where((
    (F.col('compare_name') == True) & (F.col('compare_city') == True) &
    (F.col('compare_phone-number') == True) & (F.col('compare_email') == True)
))
print(f'Records without data quality issue: {clean_data.count()} ({clean_data.count()/join_df.count()*100}%)')
dirty_data = join_df.where(~(
    (F.col('compare_name') == True) & (F.col('compare_city') == True) &
    (F.col('compare_phone-number') == True) & (F.col('compare_email') == True)
))
print(f'Records with data quality issue: {dirty_data.count()} ({dirty_data.count()/join_df.count()*100}%)')

# Result:
# Records without data quality issue: 1152 (11.52%)
# Records with data quality issue: 8848 (88.48%)
######################################################################################

######################################################################################
# name report
name_dq = join_df.where(F.col('compare_name') == False)
print(f'Data "name" with data quality issue: {name_dq.count()} ({name_dq.count()/join_df.count()*100}%)')

name_dirty_string = name_dq.select(F.explode(F.col('name_diff')).alias('dirty_string'))
name_dirty_string = name_dirty_string.groupBy('dirty_string').count()

sample_name_dirty_string = name_dq.select(
    'loyalty_name', 'transactions_name',
    F.explode(F.col('name_diff')).alias('dirty_string_al')
)
sample_name_dirty_string = sample_name_dirty_string.withColumn('rn', F.row_number().over(Window.partitionBy(F.col('dirty_string_al')).orderBy('loyalty_name')))
sample_name_dirty_string = sample_name_dirty_string.where(F.col('rn') == 1).drop('rn')

name_dq_sample_error = name_dirty_string.join(sample_name_dirty_string, name_dirty_string['dirty_string'] == sample_name_dirty_string['dirty_string_al'], how='left').drop('dirty_string_al')
print('Showing top 10 dirty data added to loyalty data with samples')
name_dq_sample_error.orderBy(F.col('count').desc()).show(10, False)

# Result:
# Data "name" with data quality issue: 4206 (42.059999999999995%)
# Showing top 10 additional characters added to attribute with samples
# +------------+-----+---------------------+-----------------+
# |dirty_string|count|loyalty_name         |transactions_name|
# +------------+-----+---------------------+-----------------+
# |/           |597  | Mar ia Cameron/     |Maria Cameron    |
# |            |584  |    Ali cia Rubio    |Alicia Rubio     |
# |#%          |571  |    Brando#%n White  |Brandon White    |
# |[           |565  |    J[ason Stewart MD|Jason Stewart MD |
# |.           |557  |    Edward .W#%alker |Edward Walker    |
# |...         |552  |    Adam V...argas   |Adam Vargas      |
# |]           |550  |    Noah Tho]mpson   |Noah Thompson    |
# |121         |536  |    K121elly Conrad  |Kelly Conrad     |
# |            |490  |    Adam V...argas   |Adam Vargas      |
# |            |8    |     Charles Jones   |Charles Jones    |
# +------------+-----+---------------------+-----------------+
# only showing top 10 rows

######################################################################################

######################################################################################
# city report
city_dq = join_df.where(F.col('compare_city') == False)
print(f'Data "city" with data quality issue: {city_dq.count()} ({city_dq.count()/join_df.count()*100}%)')

city_dirty_string = city_dq.select(F.explode(F.col('city_diff')).alias('dirty_string'))
city_dirty_string = city_dirty_string.groupBy('dirty_string').count()

sample_city_dirty_string = city_dq.select(
    'loyalty_city', 'transactions_city',
    F.explode(F.col('city_diff')).alias('dirty_string_al')
)
sample_city_dirty_string = sample_city_dirty_string.withColumn('rn', F.row_number().over(Window.partitionBy(F.col('dirty_string_al')).orderBy('loyalty_city')))
sample_city_dirty_string = sample_city_dirty_string.where(F.col('rn') == 1).drop('rn')

city_dq_sample_error = city_dirty_string.join(sample_city_dirty_string, city_dirty_string['dirty_string'] == sample_city_dirty_string['dirty_string_al'], how='left').drop('dirty_string_al')
print('Showing top 10 dirty data added to loyalty data with samples')
city_dq_sample_error.orderBy(F.col('count').desc()).show(10, False)

# Result:
# Data "city" with data quality issue: 4125 (41.25%)
# Showing top 10 dirty data added to loyalty data with samples
# +------------+-----+-----------------+-----------------+
# |dirty_string|count|loyalty_city     |transactions_city|
# +------------+-----+-----------------+-----------------+
# |[           |572  |    Me[lder      |Melder           |
# |]           |550  |    Bri    e]lle |Brielle          |
# |            |533  |    Bordea ux    |Bordeaux         |
# |...         |529  |    .Moar...k    |Moark            |
# |/           |525  |    /Cherry/     |Cherry           |
# |121         |512  |    S121epar     |Separ            |
# |.           |505  |    Disp.utanta  |Disputanta       |
# |#%          |492  | B#%oy121nton    |Boynton          |
# |            |462  |    Algoa        |Algoa            |
# |            |8    |A     e/tna[     |Aetna            |
# +------------+-----+-----------------+-----------------+
# only showing top 10 rows

######################################################################################

######################################################################################
# phone report
phone_dq = join_df.where(F.col('compare_phone-number') == False)
print(f'Data "phone-number" with data quality issue: {phone_dq.count()} ({phone_dq.count()/join_df.count()*100}%)')

phone_dirty_string = phone_dq.select(F.explode(F.col('phone-number_diff')).alias('dirty_string'))
phone_dirty_string = phone_dirty_string.groupBy('dirty_string').count()

sample_phone_dirty_string = phone_dq.select(
    'loyalty_phone-number', 'transactions_phone-number',
    F.explode(F.col('phone-number_diff')).alias('dirty_string_al')
)
sample_phone_dirty_string = sample_phone_dirty_string.withColumn('rn', F.row_number().over(Window.partitionBy(F.col('dirty_string_al')).orderBy('loyalty_phone-number')))
sample_phone_dirty_string = sample_phone_dirty_string.where(F.col('rn') == 1).drop('rn')

phone_dq_sample_error = phone_dirty_string.join(sample_phone_dirty_string, phone_dirty_string['dirty_string'] == sample_phone_dirty_string['dirty_string_al'], how='left').drop('dirty_string_al')
print('Showing top 10 dirty data added to loyalty data with samples')
phone_dq_sample_error.orderBy(F.col('count').desc()).show(10, False)

# Result:
# Data "phone-number" with data quality issue: 4202 (42.02%)
# Showing top 10 dirty data added to loyalty data with samples
# +------------+-----+---------------------+-------------------------+
# |dirty_string|count|loyalty_phone-number |transactions_phone-number|
# +------------+-----+---------------------+-------------------------+
# |/           |585  |    727-580-9/555    |727-580-9555             |
# |[           |569  |    870-409-[9396    |870-409-9396             |
# |#%          |563  |    194-193 -478#%9  |194-193-4789             |
# |.           |552  | 445-928-3.176       |445-928-3176             |
# |            |544  |    .777-531-017    0|777-531-0170             |
# |            |526  |    194-193 -478#%9  |194-193-4789             |
# |]           |523  |    481-931]-1861    |481-931-1861             |
# |...         |521  | 800-635-...8616     |800-635-8616             |
# |121         |484  |    255-311217-4288  |255-317-4288             |
# |211         |46   | 801-981211-9141     |801-981-9141             |
# +------------+-----+---------------------+-------------------------+
# only showing top 10 rows

######################################################################################

######################################################################################
# email report
email_dq = join_df.where(F.col('compare_email') == False)
print(f'Data "email" with data quality issue: {email_dq.count()} ({email_dq.count()/join_df.count()*100}%)')

email_dirty_string = email_dq.select(F.explode(F.col('email_diff')).alias('dirty_string'))
email_dirty_string = email_dirty_string.groupBy('dirty_string').count()

sample_email_dirty_string = email_dq.select(
    'loyalty_email', 'transactions_email',
    F.explode(F.col('email_diff')).alias('dirty_string_al')
)
sample_email_dirty_string = sample_email_dirty_string.withColumn('rn', F.row_number().over(Window.partitionBy(F.col('dirty_string_al')).orderBy('loyalty_email')))
sample_email_dirty_string = sample_email_dirty_string.where(F.col('rn') == 1).drop('rn')

email_dq_sample_error = email_dirty_string.join(sample_email_dirty_string, email_dirty_string['dirty_string'] == sample_email_dirty_string['dirty_string_al'], how='left').drop('dirty_string_al')
print('Showing top 10 dirty data added to loyalty data with samples')
email_dq_sample_error.orderBy(F.col('count').desc()).show(10, False)

# Result:
# Data "name" with data quality issue: 4206 (42.059999999999995%)
# Showing top 10 additional characters added to attribute with samples
# +------------+-----+---------------------+-----------------+
# |dirty_string|count|loyalty_name         |transactions_name|
# +------------+-----+---------------------+-----------------+
# |/           |597  | Mar ia Cameron/     |Maria Cameron    |
# |            |584  |    Ali cia Rubio    |Alicia Rubio     |
# |#%          |571  |    Brando#%n White  |Brandon White    |
# |[           |565  |    J[ason Stewart MD|Jason Stewart MD |
# |.           |557  |    Edward .W#%alker |Edward Walker    |
# |...         |552  |    Adam V...argas   |Adam Vargas      |
# |]           |550  |    Noah Tho]mpson   |Noah Thompson    |
# |121         |536  |    K121elly Conrad  |Kelly Conrad     |
# |            |490  |    Adam V...argas   |Adam Vargas      |
# |            |8    |     Charles Jones   |Charles Jones    |
# +------------+-----+---------------------+-----------------+
# only showing top 10 rows

######################################################################################
