!pip3 install numpy
!pip3 install pandas
!pip3 install scikit-learn
!pip3 install mlflow
!git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap
!pip3 install dill


import sys
import mlflow
import mlflow.sklearn
#mlflow.set_experiment('Retrain_exp')
experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
dfExperiments=mlflow.search_runs(experiment_ids=experimentId)
maxmetric=dfExperiments["metrics.precision"].max()
runId=dfExperiments[dfExperiments["metrics.precision"]==maxmetric].run_id

script_descriptor = open("3_trainStrategy_job.py")
a_script = script_descriptor.read()
sys.argv = ["3_trainStrategy_job.py", runId.item()]

exec(a_script)