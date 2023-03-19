################################################################################
################################################################################
## Scikit-learn DecisionTreeClassifier
## USGS Reported Seismic Event Classification Model
## Author: K. Chadwick/Nask
## Created: 02 November 2020
## Updated: 05 November 2020
##
################################################################################
## CHANGE LOG
## 2020-11-03 - Original script
## 2020-11-05 - Added Summary Report as output_string
##
################################################################################

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,roc_auc_score

################################################################################
## FUNCTIONS

def unpack(packed_dictionary):
    
    unpacked_dictionary = dict()
    for key,val in packed_dictionary.items():
        for v in val:
            unpacked_dictionary[v] = '%s{%s}' % (v,key)
            
    return unpacked_dictionary

################################################################################
## MODEL DATA

dataset_name = 'USGS Seismic Events'
input_string = '{"brew Instance": "https://demo.brewlytics.com", "Model Execution Time": "2023-03-19T03:03:31.891873Z", "brew Instance CV Type": "string", "Model Name": "Model - Scikit-learn DecisonTreeClassifier - USGS Live Data (UPDATED)", "Model Name CV Type": "string", "Model Id": "bb6f7fb0-fc7f-49ab-9595-f9fdcd863d58", "Model Id CV Type": "string", "User Name": "Mark", "User Email": "mark.wood@greymattersdefense.com", "User Phone": "----", "User ID": "abc09ea5-573e-47cf-9f91-c456f20aaf02", "User URI": "urn:brew:profiles:abc09ea5-573e-47cf-9f91-c456f20aaf02", "USGS API URL": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv", "USGS API URL CV Type": "string", "Set Random Seed": 2341, "Train/Test Split": 0.2, "Features Columns": ["latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError", "magNst"], "Target": ["type"], "Transparent Background": true}'

## Model Data
md = json.loads(input_string)

## USGS API URL
url = md['USGS API URL']

## If Set Random Seed is not set to 0, accept value; if set to 0, set to None
## to allow random set seeds between successive iterations to demonstrate the
## variability of the model
if md['Set Random Seed'] != 0:
    random_seed = md['Set Random Seed']
else:
    random_seed = None

## Set size of Test dataset to 20-percent of total dataset
test_size = md['Train/Test Split']

## Decision Tree features
features = md['Features Columns']

## USGS Seismic Event Types
target = md['Target']

## Visualization - Transparent Background
transparent = md['Transparent Background']

##------------------------------------------------------------------------------
## No Model Data -  For local testing/development

# url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'

# random_seed = 3453
# test_size = 0.2

# features = ['latitude','longitude','depth','mag','nst','gap','dmin','rms',
#             'horizontalError','depthError','magError','magNst']
# target = ['type']

# transparent = True

##------------------------------------------------------------------------------
## CV Types

packed_cv_types = {'decimal': ['latitude','longitude','depth','mag','nst',
                               'gap','dmin','rms','horizontalError',
                               'depthError','magError','magNst'],
                   'string': ['magType','net','id','place','type','status',
                              'locationSource','magSource'], 
                   'timestamp': ['time','updated']}

cv_types = unpack(packed_cv_types)
    

################################################################################
## BODY

df = pd.read_csv(url)
# df.columns = [column_name.split('{')[0] for column_name in df.columns]

## Drop rows with missing predictor values
df.dropna(subset = features, inplace = True)

##------------------------------------------------------------------------------

## Decision Tree features
X = df[features]

## Target variable
y = df[target] 

## Build Training & Test datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = test_size,
                                                 random_state = random_seed)

## Create a list of classification names
class_names = sorted(y_train[target[0]].unique())

## Create a DecisionTreeClassifier object
clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', 
                             max_depth = None, min_samples_split = 2, 
                             min_samples_leaf = 1, min_weight_fraction_leaf = 0.0,
                             max_features = None, random_state = random_seed,
                             max_leaf_nodes = None, min_impurity_decrease = 0.0,
                             class_weight = None, ccp_alpha = 0.0)

## Train the classifier
clf = clf.fit(X_train,y_train)

## Test the classifier
y_pred = clf.predict(X_test)

## Model Results metrics
result = confusion_matrix(y_test, y_pred)
result1 = classification_report(y_test, y_pred)
result2 = accuracy_score(y_test,y_pred)

##------------------------------------------------------------------------------
## Visualize DecisionTreeClassifier Performance
fig = plt.figure(figsize=(10,3), dpi = 600)

if len(class_names) > 2:
    ax = fig.add_subplot(111)
    x = 0.5
else:
    ax = fig.add_subplot(121)
    x = 1.5

## Create a Confusion Matrix
sns.heatmap(confusion_matrix(y_test,y_pred),
            annot = True,
            cmap = 'Reds')

plt.text(x,-0.4,'%s Type DecisionTreeClassifier' % dataset_name,
         fontweight = 'bold', fontsize = 12)

plt.title('Confusion Matrix', fontweight = 'bold', fontsize = 10)
plt.xlabel('Predicted Label',
          fontweight = 'normal', fontsize = 8)
plt.ylabel('Actual Label',
          fontweight = 'normal', fontsize = 8)

if len(class_names) == 2:

    ## Create an ROC/AUC Plot
    ax1 = fig.add_subplot(122)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr,label = "%.2f" % roc_auc_score(y_test,y_pred))
    plt.plot([0,1],[0,1],
             color = 'firebrick',
             linewidth = 1.0,
             linestyle = ':',
             label = "%.2f" % (0.5))
    plt.legend(loc = 'lower right')

    plt.title('ROC Curve', fontweight = 'bold', fontsize = 10)
    plt.xlabel('False positive rate (1-Specificity)',
              fontweight = 'normal', fontsize = 8)
    plt.ylabel('True positive rate (Sensitivity)',
              fontweight = 'normal', fontsize = 8)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.grid(True)

filename = 'DecisionTree_(%s).png' % md['Model Execution Time'].split('T')[0]
plt.savefig(filename, 
            dpi = 600, 
            transparent = transparent,
            bbox_inches = 'tight')
plt.show()

##------------------------------------------------------------------------------
## Add CV Types to column names

df.columns = [cv_types[column_name] for column_name in df.columns]

################################################################################
## OUTPUTS

output_table = df
output_resource = filename

################################################################################
## SUMMARY

stub = 'Scikit-learn DecisionTreeClassifier Model Details\n'
stub += 'Model Name: %s\n' % md['Model Name']
stub += 'Date: %s\n' % md['Model Execution Time']
stub += '\n'
stub += '================================================================================\n\n'
stub += 'Dataset Name: %s\n' % (dataset_name)
stub += 'Dataset size: %d rows %d columns\n\n' % (df.shape)
stub += 'Features Columns:\n- ' + '\n- '.join(features)
stub += '\n\n'
stub += 'Target/Class Columns: \n- ' + '\n- '.join(target)
stub += '\n\n'
stub += '================================================================================\n\n'
stub += 'Train/Test Information:\n\n'
stub += '- Train/Test Split: %0.2f/%0.2f\n' % (1 - test_size, test_size)
stub += '- Train Dataset size: %d rows %d columns\n' % X_train.shape
stub += '- Test Dataset size: %d rows %d columns\n' % X_test.shape
stub += '\n'
stub += 'Classes in Training Set:\n- ' + '\n- '.join(class_names)
stub += '\n\n'
stub += 'Classes in Test Set:\n- ' + '\n- '.join(sorted(y_test[target[0]].unique()))
stub += '\n\n'
stub += '================================================================================\n\n'
stub += 'Confusion Matrix:\n\n'
stub += np.array2string(result, formatter={'float_kind':lambda x: "%.2f" % x})
stub += '\n\n'
stub += 'Classification Report for Train Dataset:\n\n'
stub += result1
stub += '\n\n'
stub += 'Accuracy: %f\n\n' % result2
stub += '================================================================================\n\n'
stub += 'Features and Importance Scores:\n\n'
stub += '- ' + '\n- '.join(['%s: %f' % tuple(z) 
                            for z in zip(X_train[features], clf.feature_importances_)])
stub += '\n\n================================================================================\n\n'

output_string = stub

print(stub)
