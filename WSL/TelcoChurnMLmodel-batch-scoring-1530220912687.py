#!/usr/bin/python

import sys, os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import pandas
import dsx_core_utils, re, jaydebeapi
from sqlalchemy import *
from sqlalchemy.types import String, Boolean

# define variables
args = {'source': '/datasets/customer_churn.csv', 'output_datasource_type': '', 'target': '/datasets/customer_churn_output.txt', 'execution_type': 'DSX', 'sysparm': '', 'output_type': 'Localfile'}
input_data = os.getenv("DSX_PROJECT_DIR") + args.get("source")
output_data = os.getenv("DSX_PROJECT_DIR") + args.get("target")
model_path = os.getenv("DSX_PROJECT_DIR") + "/models/Telco_Churn_ML_model/1/model"

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# read test dataframe (inputJson = "input.json") 
testDF = SQLContext(sc).read.csv(input_data, header='true', inferSchema='true')

# load model
model_rf = PipelineModel.load(model_path)

# prediction
outputDF = model_rf.transform(testDF) 

# save scoring result to given target
scoring_df = outputDF.toPandas()

# save output to csv
scoring_df.to_csv(output_data)