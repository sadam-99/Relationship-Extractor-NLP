# Entities Relation Extraction 

# CODE AUTHOR
# SHIVAM GUPTA (NET ID: SXG19040)
# PRACHI VATS  (NET ID: PXV180021)
# Entities Relationship Extraction Project

     
## How to Use the Scripts:

## Compiling and Running the Code:

### For Preprocessing the Training and Testing Data:
Run command:- ``` python preprocessing_All_Data.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt

### For Creating the Dependency Parser data for Training Data:
Run command:- ``` python Dep_Path_Extractor.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt

### For Running the Task 2 on a given test sentense in a text file:
Run command:- ``` python Task2.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt


## For Training:
## For Training SDP-LSTM Model :
Run command:- ``` python LSTM_Train.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt

## For Testing:
## For testing SDP-LSTM Model on Testing Data:
Run command:- ``` python LSTM_Test.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt

## For prediction:
## For prediction SDP-LSTM Model on Test Sentence in a given text file:
Run command:- ``` python LSTM_Predict.py  ``` on UTD CS Linux Servers / Anaconda Prompt/Command Prompt


# Note: 
The folder named ```SDP-LSTM-Model``` contains the our Model Scripts and Checkpoints(weights meta file)

The folder named ```DATA_FILES``` contains the data files and the preprocessed data and also their Dependency Parsed Features which are used in the training of the Model

The folder named ```Checkpoint_Model``` contains the Checkpoint(Models) weights for 100th Epoch which we trained!