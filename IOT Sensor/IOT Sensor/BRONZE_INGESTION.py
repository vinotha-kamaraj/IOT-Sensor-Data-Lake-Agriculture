# Databricks notebook source
from pyspark.sql.functions import * 
BASE_PATH = "/Volumes/agriculture_iot/bronze/volume"
 
SENSOR_READINGS_PATH = f"{BASE_PATH}/sensor_readings.csv"
CROP_MASTER_PATH = f"{BASE_PATH}/crop_master.csv"
FARM_MASTER_PATH = f"{BASE_PATH}/farm_master.csv"
FIELD_MASTER_PATH = f"{BASE_PATH}/field_master.csv"
SENSOR_MASTER_PATH = f"{BASE_PATH}/sensor_master.csv"



# COMMAND ----------

sensor_readings_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(SENSOR_READINGS_PATH)
)

display(sensor_readings_df)

# COMMAND ----------

print("Sensor Reading Count:", sensor_readings_df.count())

sensor_readings_df.printSchema()

# COMMAND ----------

bronze_sensor_readings = (
    sensor_readings_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", col("_metadata.file_path"))
)

# COMMAND ----------

# DBTITLE 1,Write bronze sensor readings to Delta table
bronze_sensor_readings.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("agriculture_iot.bronze.sensor_readings")

# COMMAND ----------

sensor_master_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(SENSOR_MASTER_PATH)
)

display(sensor_master_df)

# COMMAND ----------

bronze_sensor_master = (
    sensor_master_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", col("_metadata.file_path"))
)

# COMMAND ----------

bronze_sensor_master.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("agriculture_iot.bronze.sensor_master")

# COMMAND ----------

farm_master_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(FARM_MASTER_PATH)
)

display(farm_master_df)

# COMMAND ----------

bronze_farm_master = (
    farm_master_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", col("_metadata.file_path"))
)

# COMMAND ----------

bronze_farm_master.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("agriculture_iot.bronze.farm_master")

# COMMAND ----------

crop_master_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(CROP_MASTER_PATH)
)

display(crop_master_df)

# COMMAND ----------

bronze_crop_master = (
    crop_master_df
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", col("_metadata.file_path"))
)

# COMMAND ----------

bronze_crop_master.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("agriculture_iot.bronze.crop_master")

# COMMAND ----------

# MAGIC %sql
# MAGIC select irregation where irregation="required";   