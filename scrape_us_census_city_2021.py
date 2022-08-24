from pyspark.sql import SparkSession
import requests
import csv

spark = SparkSession.builder.appName('scrape_us_census_city_2021').getOrCreate()

url = 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/cities/totals/sub-est2021_all.csv'
r = requests.get(url)
csv_content = r.content.decode('latin1')
reader = csv.reader(csv_content.splitlines(), delimiter=',')
csv_data = list(reader)

columns = csv_data[0]
data = csv_data[1:]

df = spark.createDataFrame(data, columns)
df.write.option('header', 'true').mode('overwrite').csv('enrich_dataset/us_census_city')
