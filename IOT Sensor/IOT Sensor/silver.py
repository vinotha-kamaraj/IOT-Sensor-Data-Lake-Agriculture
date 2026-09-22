# Databricks notebook source
# DBTITLE 1,Imports and Configuration
from pyspark.sql.functions import *
from pyspark.sql.window import Window

BRONZE_SENSOR_READINGS = "agriculture_iot.bronze.sensor_readings"
BRONZE_SENSOR_MASTER = "agriculture_iot.bronze.sensor_master"
BRONZE_FARM_MASTER = "agriculture_iot.bronze.farm_master"
BRONZE_CROP_MASTER = "agriculture_iot.bronze.crop_master"

SILVER_VALID_SENSOR = "agriculture_iot.silver.valid_sensor"
SILVER_INVALID_SENSOR = "agriculture_iot.silver.invalid_sensor"

# COMMAND ----------

# DBTITLE 1,Read Bronze Data
bronze_sensor_df = spark.table(
    BRONZE_SENSOR_READINGS
)

display(bronze_sensor_df)

sensor_master_df = (
    spark.table(BRONZE_SENSOR_MASTER)
    .select("sensor_id")
    .dropDuplicates()
)

display(sensor_master_df)

farm_master_df = spark.table(
    BRONZE_FARM_MASTER
)

display(farm_master_df)

crop_master_df = spark.table(
    BRONZE_CROP_MASTER
)

display(crop_master_df)

# COMMAND ----------

sensor_df = (
    bronze_sensor_df

    .withColumn("sensor_id", trim(col("sensor_id")))
    .withColumn("farm_id", trim(col("farm_id")))

    .withColumn(
        "timestamp",
        expr("try_to_timestamp(timestamp, 'dd-MM-yyyy HH:mm')")
    )

    .withColumn(
        "temperature",
        expr("try_cast(temperature AS DOUBLE)")
    )

    .withColumn(
        "humidity",
        expr("try_cast(humidity AS DOUBLE)")
    )

    .withColumn(
        "soil_moisture",
        expr("try_cast(soil_moisture AS DOUBLE)")
    )

    .withColumn(
        "rainfall",
        expr("try_cast(rainfall AS DOUBLE)")
    )

    .withColumn(
        "battery",
        expr("try_cast(battery AS DOUBLE)")
    )
)

display(sensor_df)

# COMMAND ----------

# DBTITLE 1,Data Cleaning and Standardization
sensor_df = (
    bronze_sensor_df

    .withColumn(
        "sensor_id",
        trim(col("sensor_id"))
    )

    .withColumn(
        "farm_id",
        trim(col("farm_id"))
    )

    .withColumn(
        "timestamp",
        expr("try_to_timestamp(timestamp, 'dd-MM-yyyy HH:mm')")
    )

    .withColumn(
        "temperature",
        expr("try_cast(temperature AS DOUBLE)")
    )

    .withColumn(
        "humidity",
        expr("try_cast(humidity AS DOUBLE)")
    )

    .withColumn(
        "soil_moisture",
        expr("try_cast(soil_moisture AS DOUBLE)")
    )

    .withColumn(
        "rainfall",
        expr("try_cast(rainfall AS DOUBLE)")
    )

    .withColumn(
        "battery",
        expr("try_cast(battery AS DOUBLE)")
    )
)

display(sensor_df)

# COMMAND ----------

# DBTITLE 1,Inspect Silver Data Types
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, NumericType, TimestampType

# Read your Silver table
df = spark.table(SILVER_VALID_SENSOR)

# Replace NULL values based on column datatype
for field in df.schema.fields:
    
    column_name = field.name
    
    if isinstance(field.dataType, NumericType):
        df = df.fillna({column_name: 0})
    
    elif isinstance(field.dataType, StringType):
        df = df.fillna({column_name: "UNKNOWN"})
    
    elif isinstance(field.dataType, TimestampType):
        df = df.withColumn(
            column_name,
            F.coalesce(
                F.col(column_name),
                F.current_timestamp()
            )
        )

# Overwrite the Silver table with cleaned data
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_VALID_SENSOR)
)

# Display cleaned data
display(spark.table(SILVER_VALID_SENSOR))

# COMMAND ----------

validated_sensor_df = (
    sensor_df.alias("s")
    .join(
        sensor_master_df.alias("m"),
        col("s.sensor_id") == col("m.sensor_id"),
        "left"
    )
    .withColumn(
        "sensor_id_valid",
        when(
            col("m.sensor_id").isNotNull(),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_sensor_df = (
    validated_sensor_df
    .select(
        col("s.*"),
        col("sensor_id_valid")
    )
)

display(validated_sensor_df)

# COMMAND ----------

# DBTITLE 1,DATA QUALITY VALIDATION
validated_df = (
    validated_sensor_df

    # 1. Validate timestamp
    .withColumn(
        "timestamp_valid",
        when(
            col("timestamp").isNotNull(),
            lit(True)
        ).otherwise(lit(False))
    )

    # 2. Validate temperature: -10 to 60
    .withColumn(
        "temperature_valid",
        when(
            col("temperature").between(-10, 60),
            lit(True)
        ).otherwise(lit(False))
    )

    # 3. Validate humidity: 0 to 100
    .withColumn(
        "humidity_valid",
        when(
            col("humidity").between(0, 100),
            lit(True)
        ).otherwise(lit(False))
    )

    # 4. Validate soil moisture: 0 to 100
    .withColumn(
        "soil_moisture_valid",
        when(
            col("soil_moisture").between(0, 100),
            lit(True)
        ).otherwise(lit(False))
    )

    # 5. Validate rainfall: >= 0
    .withColumn(
        "rainfall_valid",
        when(
            col("rainfall") >= 0,
            lit(True)
        ).otherwise(lit(False))
    )

    # 6. Validate battery: 0 to 100
    .withColumn(
        "battery_valid",
        when(
            col("battery").between(0, 100),
            lit(True)
        ).otherwise(lit(False))
    )

    # 7. Final validation
    .withColumn(
        "is_valid",
        col("sensor_id_valid") &
        col("timestamp_valid") &
        col("temperature_valid") &
        col("humidity_valid") &
        col("soil_moisture_valid") &
        col("rainfall_valid") &
        col("battery_valid")
    )

    # 8. Reason for invalid record
    .withColumn(
        "invalid_reason",
        concat_ws(
            "; ",
            when(~col("sensor_id_valid"), lit("INVALID_SENSOR_ID")),
            when(~col("timestamp_valid"), lit("INVALID_TIMESTAMP")),
            when(~col("temperature_valid"), lit("INVALID_TEMPERATURE_RANGE")),
            when(~col("humidity_valid"), lit("INVALID_HUMIDITY_RANGE")),
            when(~col("soil_moisture_valid"), lit("INVALID_SOIL_MOISTURE_RANGE")),
            when(~col("rainfall_valid"), lit("INVALID_RAINFALL")),
            when(~col("battery_valid"), lit("INVALID_BATTERY_RANGE"))
        )
    )
)

display(validated_df)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Sensor ID Validation
validated_sensor_df = (
    sensor_df.alias("s")
    .join(
        sensor_master_df.alias("m"),
        col("s.sensor_id") == col("m.sensor_id"),
        "left"
    )
    .withColumn(
        "sensor_id_valid",
        when(
            col("m.sensor_id").isNotNull(),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_sensor_df = (
    validated_sensor_df
    .select(
        col("s.*"),
        col("sensor_id_valid")
    )
)

display(validated_sensor_df)

# COMMAND ----------

# DBTITLE 1,Timestamp and Range Validation
validated_df = (
    validated_sensor_df
    .withColumn(
        "timestamp_valid",
        when(
            col("timestamp").isNotNull(),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "temperature_valid",
        when(
            col("temperature").between(-10, 60),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "humidity_valid",
        when(
            col("humidity").between(0, 100),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "soil_moisture_valid",
        when(
            col("soil_moisture").between(0, 100),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "rainfall_valid",
        when(
            col("rainfall") >= 0,
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "battery_valid",
        when(
            col("battery").between(0, 100),
            lit(True)
        ).otherwise(
            lit(False)
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "is_valid",
        (
            col("sensor_id_valid") &
            col("timestamp_valid") &
            col("temperature_valid") &
            col("humidity_valid") &
            col("soil_moisture_valid") &
            col("rainfall_valid") &
            col("battery_valid")
        )
    )
)

validated_df = (
    validated_df
    .withColumn(
        "invalid_reason",
        concat_ws(
            "; ",

            when(
                ~col("sensor_id_valid"),
                lit("INVALID_SENSOR_ID")
            ),

            when(
                ~col("timestamp_valid"),
                lit("INVALID_TIMESTAMP")
            ),

            when(
                ~col("temperature_valid"),
                lit("INVALID_TEMPERATURE_RANGE")
            ),

            when(
                ~col("humidity_valid"),
                lit("INVALID_HUMIDITY_RANGE")
            ),

            when(
                ~col("soil_moisture_valid"),
                lit("INVALID_SOIL_MOISTURE_RANGE")
            ),

            when(
                ~col("rainfall_valid"),
                lit("INVALID_RAINFALL")
            ),

            when(
                ~col("battery_valid"),
                lit("INVALID_BATTERY_RANGE")
            )
        )
    )
)

silver_valid_sensor_df = (
    validated_df
    .filter(
        col("is_valid") == True
    )
)

silver_invalid_sensor_df = (
    validated_df
    .filter(
        col("is_valid") == False
    )
)

silver_invalid_sensor_df = (
    silver_invalid_sensor_df
    .withColumn(
        "quarantine_timestamp",
        current_timestamp()
    )
    .withColumn(
        "quarantine_status",
        lit("QUARANTINED")
    )
)

display(silver_invalid_sensor_df)

# COMMAND ----------

# DBTITLE 1,Deduplication
dedup_window = (
    Window
    .partitionBy(
        "sensor_id",
        "farm_id",
        "timestamp"
    )
    .orderBy(
        col("_ingestion_timestamp").desc_nulls_last()
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "duplicate_rank",
        row_number().over(dedup_window)
    )
    .filter(
        col("duplicate_rank") == 1
    )
    .drop("duplicate_rank")
)


# ============================================================
# DATE AND TIME COLUMNS
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "reading_date",
        to_date(col("timestamp"))
    )
    .withColumn(
        "reading_hour",
        date_trunc(
            "hour",
            col("timestamp")
        )
    )
)


# ============================================================
# TEMPERATURE STATUS
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "temperature_status",
        when(
            col("temperature") >= 40,
            lit("HIGH")
        )
        .when(
            col("temperature") <= 5,
            lit("LOW")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)


# ============================================================
# SOIL STATUS
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "soil_status",
        when(
            col("soil_moisture") < 30,
            lit("LOW")
        )
        .when(
            col("soil_moisture") > 80,
            lit("HIGH")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)


# ============================================================
# HUMIDITY STATUS
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "humidity_status",
        when(
            col("humidity") < 30,
            lit("LOW")
        )
        .when(
            col("humidity") > 80,
            lit("HIGH")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)


# ============================================================
# IRRIGATION REQUIRED
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "irrigation_required",
        when(
            (col("soil_moisture") < 30) &
            (col("rainfall") < 2),
            lit("IRRIGATION_REQUIRED")
        )
        .otherwise(
            lit("NO_IRRIGATION_REQUIRED")
        )
    )
)


# ============================================================
# SENSOR HEALTH
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "sensor_health",
        when(
            col("battery") < 20,
            lit("LOW_BATTERY")
        )
        .when(
            col("battery") < 50,
            lit("WARNING")
        )
        .otherwise(
            lit("HEALTHY")
        )
    )
)


display(silver_valid_sensor_df)

# COMMAND ----------

# DBTITLE 1,Derived Columns and Business Rules
silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "reading_date",
        to_date(col("timestamp"))
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "reading_hour",
        date_trunc(
            "hour",
            col("timestamp")
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "temperature_status",
        when(
            col("temperature") >= 40,
            lit("HIGH")
        )
        .when(
            col("temperature") <= 5,
            lit("LOW")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "soil_status",
        when(
            col("soil_moisture") < 30,
            lit("LOW")
        )
        .when(
            col("soil_moisture") > 80,
            lit("HIGH")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "humidity_status",
        when(
            col("humidity") < 30,
            lit("LOW")
        )
        .when(
            col("humidity") > 80,
            lit("HIGH")
        )
        .otherwise(
            lit("NORMAL")
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "irrigation_required",
        when(
            (col("soil_moisture") < 30) &
            (col("rainfall") < 2),
            lit("IRRIGATION_REQUIRED")
        )
        .otherwise(
            lit("NO_IRRIGATION_REQUIRED")
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "sensor_health",
        when(
            col("battery") < 20,
            lit("LOW_BATTERY")
        )
        .when(
            col("battery") < 50,
            lit("WARNING")
        )
        .otherwise(
            lit("HEALTHY")
        )
    )
)

# COMMAND ----------

# DBTITLE 1,Time-Series Processing
sensor_window = (
    Window
    .partitionBy("sensor_id")
    .orderBy("timestamp")
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "previous_temperature",
        lag("temperature").over(
            sensor_window
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "temperature_change",
        col("temperature") -
        col("previous_temperature")
    )
)

three_reading_window = (
    Window
    .partitionBy("sensor_id")
    .orderBy("timestamp")
    .rowsBetween(-2, 0)
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "3_reading_average",
        avg("temperature").over(
            three_reading_window
        )
    )
)

daily_window = (
    Window
    .partitionBy(
        "sensor_id",
        "reading_date"
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "daily_average",
        avg("temperature").over(
            daily_window
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "daily_min",
        min("temperature").over(
            daily_window
        )
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "daily_max",
        max("temperature").over(
            daily_window
        )
    )
)

# COMMAND ----------

# DBTITLE 1,Farm Data Enrichment
farm_columns = [
    c
    for c in farm_master_df.columns
    if c != "farm_id"
]

silver_valid_sensor_df = (
    silver_valid_sensor_df.alias("s")
    .join(
        farm_master_df.alias("f"),
        col("s.farm_id") == col("f.farm_id"),
        "left"
    )
    .select(
        col("s.*"),
        *[
            col(f"f.{c}").alias(c)
            for c in farm_columns
        ]
    )
)

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "_silver_processed_timestamp",
        current_timestamp()
    )
)

display(silver_valid_sensor_df)

# COMMAND ----------

# DBTITLE 1,Additional Master Data Enrichment

# in silver_valid_sensor_df
farm_columns = [
    c
    for c in farm_master_df.columns
    if c not in silver_valid_sensor_df.columns
    and c != "farm_id"
]


# ============================================================
# JOIN SENSOR DATA WITH FARM MASTER
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df.alias("s")
    .join(
        farm_master_df.alias("f"),
        col("s.farm_id") == col("f.farm_id"),
        "left"
    )
    .select(
        col("s.*"),
        *[
            col(f"f.{c}").alias(c)
            for c in farm_columns
        ]
    )
)


# ============================================================
# ADD SILVER PROCESSING TIMESTAMP
# ============================================================

silver_valid_sensor_df = (
    silver_valid_sensor_df
    .withColumn(
        "_silver_processed_timestamp",
        current_timestamp()
    )
)


# ============================================================
# DISPLAY RESULT
# ============================================================

display(silver_valid_sensor_df)

# COMMAND ----------

# ==========================================
# CELL 14 - WRITE VALID & INVALID SILVER DATA
# ==========================================

# 1. Prepare valid records
silver_valid_sensor = (
    validated_df
    .filter(col("is_valid") == True)
    .dropDuplicates(["sensor_id", "farm_id", "timestamp"])
)

# 2. Prepare invalid records for quarantine
silver_invalid_sensor = (
    validated_df
    .filter(col("is_valid") == False)
    .withColumn(
        "quarantine_status",
        lit("QUARANTINED")
    )
    .withColumn(
        "quarantine_timestamp",
        current_timestamp()
    )
)

# 3. Write VALID records to Silver table
(
    silver_valid_sensor
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_VALID_SENSOR)
)

# 4. Write INVALID records to quarantine table
(
    silver_invalid_sensor
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_INVALID_SENSOR)
)

# 5. Count valid and invalid records
valid_count = silver_valid_sensor.count()
invalid_count = silver_invalid_sensor.count()

print("====================================")
print("SILVER DATA QUALITY SUMMARY")
print("====================================")
print(f"Valid records   : {valid_count}")
print(f"Invalid records : {invalid_count}")
print(f"Total records   : {valid_count + invalid_count}")

# 6. Display valid records
print("VALID SILVER DATA")
display(silver_valid_sensor)

# 7. Display quarantined records
print("INVALID / QUARANTINED DATA")
display(silver_invalid_sensor)

# COMMAND ----------

from pyspark.sql import functions as F

# ============================================================
# NULL VALUE HANDLING - SINGLE CELL
# ============================================================

# Use your current cleaned dataframe here
# Change "validated_df" to your dataframe name if needed

required_columns = [
    "reading_id",
    "sensor_id",
    "farm_id",
    "field_id",
    "timestamp",
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "battery"
]

# ------------------------------------------------------------
# 1. VALID DATA
#    Keep only rows where all required columns are NOT NULL
# ------------------------------------------------------------

silver_valid_sensor = validated_df.dropna(
    subset=required_columns
)


# ------------------------------------------------------------
# 2. INVALID DATA
#    Keep rows where at least one required column is NULL
# ------------------------------------------------------------

null_condition = None

for column_name in required_columns:

    condition = F.col(column_name).isNull()

    if null_condition is None:
        null_condition = condition
    else:
        null_condition = null_condition | condition


silver_invalid_nulls = (
    validated_df
    .filter(null_condition)
    .withColumn(
        "invalid_reason",
        F.lit("NULL_OR_MISSING_REQUIRED_FIELD")
    )
    .withColumn(
        "quarantine_status",
        F.lit("QUARANTINED")
    )
    .withColumn(
        "quarantine_timestamp",
        F.current_timestamp()
    )
)


# ------------------------------------------------------------
# 3. CHECK RESULT
# ------------------------------------------------------------

print("Valid records (no required NULL values):", silver_valid_sensor.count())

print("Invalid records (NULL values found):", silver_invalid_nulls.count())


# ------------------------------------------------------------
# 4. OPTIONAL: DISPLAY INVALID NULL RECORDS
# ------------------------------------------------------------

display(silver_invalid_nulls)

# COMMAND ----------

# DBTITLE 1,Cell 13
# Read current Silver table
df = spark.table(SILVER_VALID_SENSOR)

# Replace NULL values in 'temperature'
df_clean = (
    df
    .fillna(0.0, subset=["temperature"])
)

# Overwrite the existing Silver table
(
    df_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_VALID_SENSOR)
)

# Check result
display(spark.table(SILVER_VALID_SENSOR))

# COMMAND ----------

(
    silver_invalid_sensor_df
    .write
    .format("delta")
    .mode("append")
    .option(
        "mergeSchema",
        "true"
    )
    .saveAsTable(
        SILVER_INVALID_SENSOR
    )
)

print(
    "Valid Sensor Records:",
    silver_valid_sensor_df.count()
)

print(
    "Invalid Sensor Records:",
    silver_invalid_sensor_df.count()
)

display(silver_invalid_sensor_df)

display(
    spark.table(
        SILVER_VALID_SENSOR
    )
)

display(
    spark.table(
        SILVER_INVALID_SENSOR
    )
)