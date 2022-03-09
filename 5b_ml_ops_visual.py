## Part 7b - Model Operations - Visualising Model Metrics

# This is a continuation of the previous process started in the 
# `7a_ml_ops_simulations.py` script.
# Here we will load in the metrics saved to the model database in the previous step 
# into a Pandas dataframe, and display different features as graphs. 

#```python
# help(cdsw.read_metrics)
# Help on function read_metrics in module cdsw:
#
# read_metrics(model_deployment_crn=None, start_timestamp_ms=None, end_timestamp_ms=None, model_crn=None, model_build_crn=None)
#    Description
#    -----------
#    
#    Read metrics data for given Crn with start and end time stamp
#    
#    Parameters
#    ----------
#    model_deployment_crn: string
#        model deployment Crn
#    model_crn: string
#        model Crn
#    model_build_crn: string
#        model build Crn
#    start_timestamp_ms: int, optional
#        metrics data start timestamp in milliseconds , if not passed
#        default value 0 is used to fetch data
#    end_timestamp_ms: int, optional
#        metrics data end timestamp in milliseconds , if not passed
#        current timestamp is used to fetch data
#    
#    Returns
#    -------
#    object
#        metrics data
#```
 

import cdsw, time, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import sqlite3



## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.



# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
model_id = pd.DataFrame(cml.get_models(params))['id'].max()

latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"])
metrics_df.tail().T

# Write the data to SQL lite for Viz Apps
if not(os.path.exists("model_metrics.db")):
  conn = sqlite3.connect('model_metrics.db')
  metrics_df.to_sql(name='model_metrics', con=conn)

# Do some conversions & calculations
metrics_df['startTimeStampMs'] = pd.to_datetime(metrics_df['startTimeStampMs'], unit='ms')
metrics_df['endTimeStampMs'] = pd.to_datetime(metrics_df['endTimeStampMs'], unit='ms')
metrics_df["processing_time"] = (metrics_df["endTimeStampMs"] - metrics_df["startTimeStampMs"]).dt.microseconds * 1000

# This shows how to plot specific metrics.
sns.set_style("whitegrid")
sns.despine(left=True,bottom=True)

#prob_metrics = metrics_df.dropna(subset=['metrics.probability']).sort_values('startTimeStampMs')
#sns.lineplot(x=range(len(prob_metrics)), y="metrics.probability", data=prob_metrics, color='grey')

#time_metrics = metrics_df.dropna(subset=['processing_time']).sort_values('startTimeStampMs')
#sns.lineplot(x=range(len(prob_metrics)), y="processing_time", data=prob_metrics, color='grey')

# This shows how the model accuracy drops over time.
agg_metrics = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')
sns.barplot(x=list(range(1,len(agg_metrics)+1)), y="metrics.accuracy", color="grey", data=agg_metrics)
plt.savefig('accuracyEvolution.png')


latest_aggregate_metric = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')[-1:]["metrics.accuracy"]


import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

  
from matplotlib.patches import Circle, Wedge, Rectangle

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
  
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation
  

  
def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title='', fname=False): 
    
    """
    some sanity checks first
    
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    ax.text(0, -0.05, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=22, fontweight='bold')

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200)

        
        
        
       
  

a=latest_aggregate_metric.to_list()[0]
valores=np.linspace(0,1,6)
import bisect
flecha=bisect.bisect_left(valores, a)

gauge(labels=['CRITICAL','LOW','NOT GOOD','CARE','GOOD'], colors=['#ED1C24','#F18517','#FFCC00','#0063BF','#007A00'], arrow=flecha, title='accuracy ='+str(a)) 
plt.savefig('lastAccuracyValue.png')

if a < float(cml.get_environment_variables()['THRESHOLD']):
  print("Se va a reentrenar el modelo, recibirá un correo cuando esté puesto en producción")
else:
  print("El modelo satisface los criterios establecidos")