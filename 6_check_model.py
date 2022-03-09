# # Check Model
# This file should be run in a job that will periodically check the current model's accuracy and trigger the 
# model retrain job if its below the required thresh hold. 

import cdsw, time, os
import pandas as pd
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap



# replace this with these values relevant values from the project
!pip3 install seaborn
exec(open("5a_ml_ops_simulation.py").read())

# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  



project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
model_id = pd.DataFrame(cml.get_models(params))['id'].min()
latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"])

latest_aggregate_metric = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')[-1:]["metrics.accuracy"]
#exec(open("7b_ml_ops_visual.py").read())
threshold =0.5
threshold_environment_params = {"THRESHOLD":threshold}
threshold_environment = cml.create_environment_variable(threshold_environment_params)


for job in cml.get_jobs():
  if job['name'] == "avisoPerformance":
    job_id = job['id']
cml.start_job(job_id,{})

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
for job in cml.get_jobs():
  if job['name'] == "retrain":
    job_id = job['id']
if latest_aggregate_metric.to_list()[0] < threshold:
  print("model is below threshold, retraining")
  cml.start_job(job_id,{})
  #TODO reploy new model
else:
  print("model does not need to be retrained")

  
