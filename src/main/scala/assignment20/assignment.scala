package assignment20

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}
import org.apache.spark.ml.feature.{StringIndexer, MinMaxScaler}


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)

  // Initiate the spark session.
  val spark = SparkSession.builder()
    .appName("Assignment")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  // Read 2 dimensional data.
  val dataK5D2 =  spark.read
    .option("header", "true")
    .option("sep", ",")
    .option("inferSchema", "true")
    .csv("data/dataK5D2.csv")

  // Read 3 dimensional data.
  val dataK5D3 =  spark.read
    .option("header", "true")
    .option("sep", ",")
    .option("inferSchema", "true")
    .csv("data/dataK5D3.csv")

  // Initiate StringIndexer instance.
  val indexer = new StringIndexer()
    .setInputCol("LABEL")
    .setOutputCol("num_label")

  // Transform label column to a numeric scale.
  val dataK5D3WithLabels = indexer.fit(dataK5D2).transform(dataK5D2)


  // Task #1
  /* Parameters: df, DataFrame: 2 dimensional data;
   * 							k, Int: The amount of clusters;
   * Return: Array[(Double, Double)]: Cluster means in an array;
   */
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // Initiate a pipeline with vector assembler and minmaxscaler, then transform 2D features to a vector form.
    val assembler = new VectorAssembler().setInputCols( Array("a","b") ).setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
    val pipeline = new Pipeline().setStages( Array(assembler, scaler) )

    val feature_df  = pipeline.fit(df).transform(df)

    // Initiate KMeans instance and fit the feature data.
    val km = new KMeans().setK(k).setFeaturesCol("scaled_features")
    val kmModel = km.fit(feature_df)

    // Map the cluster centers.
    val centers = kmModel.clusterCenters.map( x => ( x(0), x(1) ) )

    centers

  }

  // Task #2
  /* Parameters: df, DataFrame: 3 dimensional data;
   * 							k, Int: The amount of clusters;
   * Return: Array[(Double, Double)]: Cluster means in an array;
   */
  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    // Initiate pipeline with vector assembler and minmaxscaler, then transform 3D features to a vector form.
    val assembler = new VectorAssembler().setInputCols( Array("a","b","c") ).setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
    val pipeline = new Pipeline().setStages(Array(assembler, scaler))
    val feature_df  = pipeline.fit(dataK5D3).transform(dataK5D3)

    // Initiate KMeans instance and fit the feature data.
    val km = new KMeans().setK(k).setFeaturesCol("scaled_features")
    val kmModel = km.fit(feature_df)

    // Map the cluster centers.
    val centers = kmModel.clusterCenters.map( x => ( x(0), x(1), x(2) ) )

    centers
  }

  // Task #3
  /* Parameters: df, DataFrame: 2 dimensional data;
   * 							k, Int: The amount of clusters;
   * Return: Array[(Double, Double)]: Cluster means in an array;
   */
  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {

    // Initiate vector assembler and transform 3D features to a vector form.
    val assembler = new VectorAssembler().setInputCols( Array("a","b","num_label") ).setOutputCol("features")

    // Using scaler here made the test fail so we left it out.

    val feature_df  = assembler.transform(df)

    // Initiate KMeans instance and fit the feature data.
    val km = new KMeans().setK(k).setFeaturesCol("features")
    val kmModel = km.fit(feature_df)

    // Initiate, map and return cluster centers which have big possibility for the fatal condition.
    val centers = kmModel.clusterCenters
    val filtered = centers.filter( x => x(2) >= 0.5 )

    filtered.map(x => ( x(0), x(1) ) )
  }

  // Task #4
  /* Parameters: df, DataFrame: 2 dimensional data;
   * Return: Array[(Int, Double)]: Array of the cost of each k-value;
   */
  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    // Initiate pipeline and transform 2D features to a vector form.
    val assembler = new VectorAssembler().setInputCols(Array("a","b")).setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
    val pipeline = new Pipeline().setStages(Array(assembler, scaler))
    val feature_df  = pipeline.fit(dataK5D2).transform(dataK5D2)

    //
    val costs = new Array[(Int, Double)]( high - low + 1 )

    // Loop through the given range of k-values.
    // Initiate each new instance, fit the feature data and calculate the cost. 
    for( k <- ( low.to(high) ) ) {
      val km = new KMeans().setK(k).setFeaturesCol("features")
      val kmModel = km.fit(feature_df)
      val cost = kmModel.computeCost(feature_df)

      costs( k - low ) = ( k, cost )
    }

    costs
  }

}