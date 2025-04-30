from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

def consume_from_kafka(spark = SparkSession) -> None:
    df = (spark 
        .readStream 
        .format("kafka")
        .option("kafka.bootstrap.servers", "broker:29092") 
        .option("subscribe", "traffic_data") 
        .load()
    )
    return df


def load_files(path:str):
    df = spark.read.parquet(path)
    return df


def batch_processing_pivot(df, epoch_id) -> DataFrame: 
    df = (df.select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
             .select("data.site_id", "data.measurement_time", "data.vehicle_flow_rate"))

    df = df.dropDuplicates(['site_id', 'measurement_time'])
    df_pivot = df.groupBy("site_id").pivot("measurement_time").agg(F.first("vehicle_flow_rate"))
    df_pivot = df_pivot.orderBy("site_id").select(
        "site_id", *sorted(df_pivot.columns[1:])) 

    numeric_cols = df_pivot.columns[1:]  
    df_pivot = df_pivot.fillna({col: df_pivot.select(F.mean(col)).first()[0] for col in numeric_cols})
    df_pivot = df_pivot.na.fill(0)

    print(f"..writing data for epoch_id {epoch_id}..")
    df_pivot.repartition(1).write.mode("overwrite").parquet(f"../data/traffic_data_{epoch_id}")

    ##### Start to merge previously stored data as a window #####
    window_param = 3
    if epoch_id > window_param:
        for i in range(epoch_id - window_param, epoch_id + 1):
            path = f"../data/traffic_data_{i}"
            temp = load_files(path)  
            if i == epoch_id - window_param:
                merged_df = temp
            else:
                merged_df = merged_df.join(temp, on="site_id", how="outer")

        numeric_cols = merged_df.columns[1:] 
        merged_df = merged_df.fillna({col: merged_df.select(F.mean(col)).first()[0] for col in numeric_cols})
        merged_df = merged_df.na.fill(0)
        merged_df = merged_df.orderBy("site_id")
        sorted_columns = ["site_id"] + sorted(merged_df.columns[1:])
        merged_df = merged_df.select(*sorted_columns)

        print(f"..writing merged data for epoch_id {epoch_id-window_param}..")
        merged_df.repartition(1).write.mode("overwrite").parquet(f"../data/traffic_data_to_model_{epoch_id-window_param}")


def batch_processing(df, epoch_id) -> DataFrame:
    df = (df.select(F.from_json(F.col("value").cast("string"), schema).alias("data")).select("data.*"))
    df = df.withColumn(
        "data_quality_categorical",
        F.when(F.col("data_quality")=="good", 1).otherwise(0)
        )
    
    df = df.drop("modified_time")
    avg_vehicle_flow_rate = df.groupBy("measurement_time").agg(F.mean("vehicle_flow_rate").alias("avg_vehicle_flow_rate"))
    std_vehicle_flow_rate = df.groupBy("measurement_time").agg(F.std("vehicle_flow_rate").alias("std_vehicle_flow_rate"))

    df = df.join(avg_vehicle_flow_rate, on="measurement_time", how="left")
    df = df.join(std_vehicle_flow_rate, on="measurement_time", how="left")   
    df = df.orderBy(F.col("measurement_time")).orderBy(F.col("site_id"))

    print(f"..writing data for epoch_id {epoch_id}..")
    df.repartition(1).write.mode("overwrite").parquet(f"../data/traffic_data_{epoch_id}")

def batch_processing_update(df, epoch_id) -> DataFrame:
    df = (df.select(F.from_json(F.col("value").cast("string"), schema).alias("data")).select("data.*"))
    df = df.withColumn("data_quality_categorical", F.when(F.col("data_quality") == "good", 1).otherwise(0))
    df = df.drop("modified_time")

    avg_vehicle_flow_rate = df.groupBy("measurement_time").agg(F.mean("vehicle_flow_rate").alias("avg_vehicle_flow_rate"))
    std_vehicle_flow_rate = df.groupBy("measurement_time").agg(F.std("vehicle_flow_rate").alias("std_vehicle_flow_rate"))

    df = df.join(avg_vehicle_flow_rate, on="measurement_time", how="left")
    df = df.join(std_vehicle_flow_rate, on="measurement_time", how="left")
    df = df.orderBy(F.col("measurement_time")).orderBy(F.col("site_id"))

    ##### Load the existing DataFrame if it exists #####
    try:
        existing_df = spark.read.parquet("../data/traffic_data") 
        df = existing_df.unionByName(df).dropDuplicates(["site_id", "measurement_time"])
        df = df.orderBy(F.col("measurement_time"), F.col("site_id"))
    except Exception as e:
        print("No existing data found, creating new storage.")

    print(f"..updating data for epoch_id {epoch_id}..")
    df.repartition(1).write.mode("overwrite").parquet("../data/traffic_data")


if __name__ == "__main__":
    spark = (SparkSession.builder
                .appName("kafka_consumer")
                .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2")
                .getOrCreate())

    df_stream = consume_from_kafka(spark)

    ##### Define schema #####
    schema = StructType([
        StructField("specific_lane", StringType(), True),
        StructField("geometry", StringType(), True),
        StructField("data_quality", StringType(), True),
        StructField("vehicle_flow_rate", IntegerType(), True),
        StructField("average_vehicle_speed", FloatType(), True),
        StructField("measurement_time", TimestampType(), True),
        StructField("modified_time", TimestampType(), True),
        StructField("site_id", IntegerType(), True)
    ])

    ##### Perform the query to batch-process 5 observations using ´batch_processing´ #####
    query = (df_stream
        .writeStream
        .foreachBatch(batch_processing_update)
        .trigger(processingTime='5 minutes')
        .start()
        )

    query.awaitTermination()