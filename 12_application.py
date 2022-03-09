# Part 6: Application

# This script explains how to create and deploy Applications in CML.
# This feature allows data scientists to **get ML solutions in front of stakeholders quickly**,
# including business users who need results fast.
# This may be good for sharing a **highly customized dashboard**, a **monitoring tool**, or a **product mockup**.

# CML is agnostic regarding frameworks.
# [Flask](https://flask.palletsprojects.com/en/1.1.x/),
# [Dash](https://plotly.com/dash/),
# or even [Tornado](https://www.tornadoweb.org/en/stable/) apps will all work.
# R users will find it easy to deploy Shiny apps.

# If you haven't yet, run through the initialization steps in the README file. Do that
# now

# This file is provides a sample Flask app script, ready for deployment,
# which displays churn predictions and explanations using the Model API deployed in
# Part 5

# Deploying the Application
#
# > Once you have written an app that is working as desired, including in a test Session,
# > it can be deployed using the *New Application* dialog in the *Applications* tab in CML.

# After accepting the dialog, CML will deploy the application then *assign a URL* to
# the Application using the subdomain you chose.
#
# *Note:* This does not requirement the `cdsw-build.sh* file as it doen now follows a
# seperate build process to deploy an application.
#

# To create an Application using our sample Flask app, perform the following.
# This is a special step for this particular app:
#
# In the deployed Model from step 5, go to *Model* > *Settings* in CML and make a note (i.e. copy) the
# "**Access Key**". eg - `mqc8ypo...pmj056y`
#
# While you're there, **disable** the additional Model authentication feature by unticking **Enable Authentication**.
#
# **Note**: Disabling authentication is only necessary for this Application to work.
# Ordinarily, you may want to keep Authentication in place.
#
# Next, from the Project level, click on *Open Workbench* (note you don't actually have to Launch a
# Session) in order to edit a file. Select the `flask/single_view.html` file and paste the Access
# Key in at line 19.
#
# `        const accessKey = "mp3ebluylxh4yn5h9xurh1r0430y76ca";`
#
# Save the file (if it has not auto saved already) and go back to the Project.
#
# Finally, go to the *Applications* section of the Project and select *New Application* with the following:
# * **Name**: Churn Analysis App
# * **Subdomain**: churn-app _(note: this needs to be unique, so if you've done this before,
# pick a more random subdomain name)_
# * **Script**: 6_application.py
# * **Kernel**: Python 3
# * **Engine Profile**: 1 vCPU / 2 GiB Memory
#
# Accept the inputs, and in a few minutes the Application will be ready to use.

# Using the Application

# >  A few minutes after deploying, the *Applications* page will show the app as Running.
# You can then click on its name to access it.
# CML Applications are accessible by any user with read-only (or higher) access to the project.
#

# This deploys a basic flask application for serving the HTML and some specific data
# use for project Application.

# At this point, you will be able to open the Churn Analysis App.
# The initial view is a table of randomly selected customers from the dataset.
# This provides a snapshot of the customer base as a whole.
# The colors in the *Probability* column correspond to the prediction, with red customers being deemed more likely to churn.
# The colors of the features show which are most important for each prediction.
# Deeper red indicates incresed importance for predicting that a customer **will churn**
# while deeper blue indicates incresed importance for predicting that a customer **will not**.
#
from flask import Flask, send_from_directory, request
from IPython.display import Javascript, HTML
import random
import os
from churnexplainer import ExplainedModel
from collections import ChainMap
from flask import Flask
from pandas.io.json import dumps as jsonify
import logging
import subprocess
from IPython.display import Image
Image("images/table_view.png")
#
# Clicking on any row will show a "local" interpreted model for that particular customer.
# Here, you can see how adjusting any one of the features will change that customer's churn prediction.
#
Image("images/single_view_1.png")
#
# Changing the *InternetService* to *DSL* lowers the probablity of churn.
# **Note**: this obviously does *not* mean that you should change that customer's internet service to DSL
# and expect they will be less likely to churn.
# Imagine if your ISP did that to you.
# Rather, the model is more optimistic about an otherwise identical customer who has been using DSL.
# This information simply gives you a clearer view of what to expect given specific factors
# as a starting point for developing your business strategies.
# Furthermore, as you start implementing changes based on the model, it may change customers' behavior
# so that the predictions stop being reliable.
# It's important to use Jobs to keep models up-to-date.
#
Image("images/single_view_2.png")
#
# There are many frameworks that ease the development of interactive, informative webapps.
# Once written, it is straightforward to deploy them in CML.


# This reduces the the output to the console window
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Since we have access in an environment variable, we want to write it to our UI
# Change the line in the flask/single_view.html file.
if os.environ.get('SHTM_ACCESS_KEY') != None:
  access_key = os.environ.get('SHTM_ACCESS_KEY', "")
  subprocess.call(["sed", "-i",  's/const\saccessKey.*/const accessKey = "' +
                   access_key + '";/', "/home/cdsw/flask/single_view.html"])

from collections import ChainMap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from churnexplainer import ExplainedModel
import pickle
from pandas.io.json import json_normalize
from churnexplainer import CategoricalEncoder
import cdsw, numpy
filename="./models/champion.pkl"
#Load the model save earlier.
loaded_model = pickle.load(open(filename, 'rb'))
filename="./models/champion/ce.pkl"
#Load the model save earlier.
ce = pickle.load(open(filename, 'rb'))
df = pd.read_csv('/home/cdsw/raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv')

# *Note:* If you want to test this in a session, comment out the line 
#`@cdsw.model_metrics` below. Don't forget to uncomment when you
# deploy, or it won't write the metrics to the database 
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))

# This is the main function used for serving the model. It will take in the JSON formatted arguments , calculate the probablity of 
# churn and create a LIME explainer explained instance and return that as JSON.
def explainid(N):
    data = dataid(N)
    data = json_normalize(data)
    # This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
    data = data[[c for c, _ in cols]]
    catcols = (c for c, iscat in cols if iscat)
    for col in catcols:
      data[col] = pd.Categorical(data[col])
    

    X = ce.fit_transform(data)
    probability = loaded_model.predict(X)
      
    
    return {'data': dict(data.iloc[0]),
            'probability': probability[0],
            'explanation': 1,
            'id': int(N)}


# Gets the rest of the row data for a particular customer.


def dataid(N):
    df = pd.read_csv('/home/cdsw/raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv')
    customer_id = df.index.dtype.type(N)
    data = df.loc[[customer_id]].reset_index()
    return data.to_dict(orient='records')


# Flask doing flasky things
flask_app = Flask(__name__, static_url_path='')


@flask_app.route('/')
def home():
    return "<script> window.location.href = '/flask/table_view.html'</script>"


@flask_app.route('/flask/<path:path>')
def send_file(path):
    return send_from_directory('flask', path)

# Grabs a sample explained dataset for 10 randomly selected customers.


@flask_app.route('/sample_table')
def sample_table():
    sample_ids = random.sample(range(1, len(df)), 10)
    sample_table = []
    for ids in sample_ids:
        sample_table.append(explainid(str(ids)))
    return jsonify(sample_table)
filename="./models/champion/ce.pkl"
#Load the model save earlier.
ce = pickle.load(open(filename, 'rb'))
# Shows the names and all the catagories of the categorical variables.
def categories2():
        return {feature: list(ce.classes_[feature])
                for feature in ce.cat_columns_}

@flask_app.route("/categories")
def categories():
    return jsonify({feat: dict(enumerate(cats))
                    for feat, cats in categories2.items()})

# Shows the names and all the statistical variations of the numerica variables.
def describe(s):
            return {'median': s.median(),
                    'mean': s.mean(),
                    'min': s.min(),
                    'max': s.max(),
                    'std': s.std()}
  



@flask_app.route("/stats")
def stats(df):
  stats = {ce.non_cat_columns_[i]: describe(pd.to_numeric(df[str(ce.non_cat_columns_[i])],errors='coerce')) for i in range(len(ce.non_cat_columns_))}
  return stats


# A handy way to get the link if you are running in a session.
HTML("<a href='https://{}.{}'>Open Table View</a>".format(
    os.environ['CDSW_ENGINE_ID'], os.environ['CDSW_DOMAIN']))

# Launches flask. Note the host and port details. This is specific to CML/CDSW
if __name__ == "__main__":
    flask_app.run(host='127.0.0.1', port=int(os.environ['CDSW_APP_PORT']))
