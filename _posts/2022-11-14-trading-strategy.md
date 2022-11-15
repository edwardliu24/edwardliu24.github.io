# Trading Strategy on Chinese Stock Index Devleoped by Deep Learning

## Main Idea

In this project, we will first develop neural networks to predict the index of stock market ( In this project, I use the data of Chinese stcok index). Then based on the predictions we get, we generate trading signals that instruct the investor to long or short. Finally we back test the trading signal and inveestigate its return.

## Model Preparation

### Modules import

First, we need to import all the modules we need for this project.


```python
import math
import pandas_datareader as web
import numpy as np
from numpy  import array
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import csv
from google.colab import files
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import datetime as dt
```

###Data import and clean-up

Import the data that the investor wants to predict, in this project, use the 12 year Chinese stock index data


```python
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['AIndexEodPrices.csv']),index_col=0)
del df['S_INFO_WINDCODE'] # delete the column we don't need
```



     <input type="file" id="files-7efaa5e4-5d0e-4850-8754-04954c500ba3" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-7efaa5e4-5d0e-4850-8754-04954c500ba3">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving AIndexEodPrices.csv to AIndexEodPrices.csv
    


```python
# Save the first column as trade date
# Save the next five columns as data

training_set = df.iloc[:len(df)//4*3 , 1:6].values  # Use first 75% data for training
test_set = df.iloc[len(df)//4*3:, 1:6].values # Use last 25% data for testing
```


```python
sc  = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled  = sc.transform(test_set) 
n_future=1
n_past =30
```


```python
def data_split(data, n_future, n_past):
    X = []
    y = []
    for i in range(n_past, len(data) - n_future +1):
      X.append(data[i - n_past:i, 0:5])
      y.append(data[i + n_future - 1:i + n_future,3])
    return np.array(X), np.array(y)
```


```python
X_train, y_train = data_split(training_set_scaled, n_future,n_past)
X_train          = X_train.reshape(X_train.shape[0], X_train.shape[1], 5)

X_test, y_test   = data_split(testing_set_scaled, n_future,n_past)
X_test           = X_test.reshape(X_test.shape[0], X_test.shape[1], 5)
```


```python
model1 = keras.models.Sequential([
     layers.Input((n_past, 5)),
     layers.Reshape((n_past, 5, 1)),
     layers.Conv2D(filters=64,
                           kernel_size=3,
                           strides=1,
                           padding="same",
                           activation="relu"),
    layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
    layers.Dropout(0.3),
    layers.Reshape((n_past, -1)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation="relu"),
    layers.Dense(y_train.shape[1])
])
```


```python
model1.compile(optimizer='adam', loss='mean_squared_error')
history1 = model1.fit(X_train, y_train, batch_size=1, epochs=10)
```

    Epoch 1/10
    2310/2310 [==============================] - 27s 7ms/step - loss: 0.0020
    Epoch 2/10
    2310/2310 [==============================] - 17s 7ms/step - loss: 0.0013
    Epoch 3/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 0.0012
    Epoch 4/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 9.6511e-04
    Epoch 5/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 8.9721e-04
    Epoch 6/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 8.9269e-04
    Epoch 7/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 8.3063e-04
    Epoch 8/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 9.1967e-04
    Epoch 9/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 7.7931e-04
    Epoch 10/10
    2310/2310 [==============================] - 16s 7ms/step - loss: 8.3933e-04
    


```python
predicted_stock_price = model1.predict(X_test)                      
scaled_prediction = predicted_stock_price * (sc.data_max_[2] - sc.data_min_[2]) + sc.data_min_[2]
scaled_true = y_test * (sc.data_max_[2] - sc.data_min_[2]) + sc.data_min_[2]
```

    24/24 [==============================] - 0s 4ms/step
    


```python
train = df[:len(df)//4*3+30]
valid = df[len(df)//4*3+30:]

valid['Predictions'] = scaled_prediction
#Visualize the data
plt.figure(figsize=(160,80))
plt.title('Model')
plt.plot(train['S_DQ_CLOSE'])
plt.plot(valid[['S_DQ_CLOSE', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    


    
![png](output_17_1.png)
    



```python
def returns(days,df,model):
  balance=[0]
  signal = pd.DataFrame()
  data=df.iloc[:,1:6].values
  scaler  = MinMaxScaler(feature_range=(0, 1))
  scaled_data=scaler.fit_transform(data)
  prediction_hist=[]
  profit = []
  datelist_train = list(df['TRADE_DT'][-days - 1:])
  datelist_train = [dt.datetime.strptime(str(date), '%Y%m%d').date() for date in datelist_train]

  # return strategy
  for i in range(days):
    x=[]
    x.append(scaled_data[i:i+30, 0:5])
    x=np.array(x)
    x=x.reshape(x.shape[0],x.shape[1],5)
    prediction=model.predict(x, verbose = False)
    prediction = prediction * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]
    prediction_hist.append(prediction)
    if prediction>1.015*data[i+29][3]:
      profit.append(data[i+30][3]-data[i+29][3])
      balance.append((profit[i]+balance[i]))
      tmp = pd.DataFrame(["Long"])
      tmp.index = [datelist_train[i + 1]]
      signal = signal.append(tmp)
    elif prediction < 0.985*data[i+29][3]:
      profit.append(data[i+29][3]-data[i+30][3])
      balance.append(profit[i] + balance[i])
      tmp = pd.DataFrame(["Short"])
      tmp.index = [datelist_train[i + 1]]
      signal = signal.append(tmp)
    else:
      profit.append(0)
      balance.append(balance[i])
    
    
    # long strategy
  stock_price = df.iloc[29:,4:5].values
  long_return = [0]
  for i in range(days):
    long_return.append(long_return[i] + stock_price[i + 1] - stock_price[i])
  #print(profit)
  #print(prediction_hist)
  #print(stock_price)
  #plt.plot(np.array(prediction_hist).reshape(days,1),c='blue')
  #plt.plot(stock_price[1:,],c='red')
  signal.columns = ["Action"]
  balance = pd.DataFrame(balance)
  balance.index = datelist_train
  plt.plot(balance,c='blue')
  long_return = pd.DataFrame(long_return)
  long_return.index = datelist_train
  plt.plot(long_return, c = 'red')
  return signal, balance[days:]

```


```python
returns(365,df[-395:],model1)
```





  <div id="df-5a2e60e0-aeea-408e-aec4-ebf000f0bd95">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-05-12</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2021-05-13</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2021-05-17</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2021-05-18</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2021-05-19</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-11-01</th>
      <td>Long</td>
    </tr>
    <tr>
      <th>2022-11-03</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2022-11-07</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2022-11-08</th>
      <td>Short</td>
    </tr>
    <tr>
      <th>2022-11-09</th>
      <td>Short</td>
    </tr>
  </tbody>
</table>
<p>118 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5a2e60e0-aeea-408e-aec4-ebf000f0bd95')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5a2e60e0-aeea-408e-aec4-ebf000f0bd95 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5a2e60e0-aeea-408e-aec4-ebf000f0bd95');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





    
![png](output_19_1.png)
    
