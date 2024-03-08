import pandas as pd
import warnings

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

pd.options.mode.copy_on_write = True
warnings.simplefilter(action='ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManhattanLSTM(torch.nn.Module):
    """
    Implementing the architecture of Manhattan-LSTM

    Parameters:
    - hidden_size (int): Number of features in the hidden state
    - input_size (int): Number of features in the input
    - num_layers (int): Number of LSTM layers 
    """

    def __init__(self, hidden_size, input_size, num_layers):
        super(ManhattanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, x1, x2):

        x1, x2 = x1.view(x1.shape[0], x1.shape[1], 1), x2.view(x2.shape[0], x2.shape[1], 1)

        _, (h1, _) = self.lstm(x1)
        _, (h2, _) = self.lstm(x2)
        
        h1 = h1[-1, :, :]
        h2 = h2[-1, :, :]
        
        manhattan_dist = torch.sum(torch.abs(h1 - h2), dim=1)
        output = torch.exp(-manhattan_dist)

        return output
    

def generate_hyperparameters(n_samples):
    """
    Generate a list of parameters according to the method explained in the article section 5.3.2.

    Parameters:
    - n_samples (int): Number of samples to generate

    Returns:
    - params_list (List of dict): List containing different sets of params
    """
    params_list = []

    for n in range(n_samples):
        hidden_size = np.random.choice(np.arange(8, 136, 8))
        clip_gradient_value = np.random.choice(np.arange(0.0, 0.51, 0.01))
        batch_size = np.random.choice(np.arange(16, 528, 16))
        lr = np.random.choice(np.arange(2*10e-6, 10e-4+10e-6, 10e-6))

        params = {'hidden_size':hidden_size,
                'clip_gradient_value':clip_gradient_value,
                'batch_size':batch_size,
                'lr':lr}
        params_list.append(params)

    return params_list


def create_train_dataset(df):
    """
    Create train dataset for the task by computing new features and selecting relevant data.

    Parameters:
    - df (pandas DataFrame): Official training dataset unchanged downloaded from the competition site.

    Returns:
    - X_train (pandas DataFrame): Train data
    - y_train (pandas DataFrame): Train targets
    """

    # Keeping valid events only
    df_valid_events = df.groupby('event_id').filter(lambda x: ((x['time_to_tca'].min() <= 2.00) & (len(x[x['time_to_tca'] >= 2.0]) >= 1)))
    
    # Extracting the last CMD available for each event for target definition
    y_train = df_valid_events.loc[df_valid_events.groupby('event_id')['time_to_tca'].idxmin()][['event_id', 'risk']]

    # Extracting the training data associated to the target
    X_train = df_valid_events[df_valid_events['time_to_tca'] >= 2.0]

    # Adding the new features 
    grouped_data = X_train[['event_id', 'risk']].groupby(by='event_id')

    number_cdms = grouped_data.size()
    mean_risk_cdms = grouped_data.mean()
    std_risk_cdms = grouped_data.std().fillna(0)

    new_features = pd.DataFrame({
        'number_CDMs': number_cdms,
        'mean_risk_CDMs': mean_risk_cdms['risk'],
        'std_risk_CDMs': std_risk_cdms['risk']
    })

    X_train = pd.merge(X_train, new_features, on='event_id')

    # Keeping only the last CDM available (with time_to_tca >= 2)
    X_train = X_train.loc[X_train.groupby('event_id')['time_to_tca'].idxmin()]

    y_train.reset_index(drop=True, inplace=True) 
    X_train.reset_index(drop=True, inplace=True)

    # Computing the different classes of data
    y_train['is_anomalous_hr'] = ((X_train['risk'] < -6) & (y_train['risk'] >= -6)).astype(int)
    y_train['is_non_anomalous_lr'] = ((X_train['risk'] < -6) & (y_train['risk'] < -6)).astype(int)

    # Picking relevant features only
    X_train = X_train[['time_to_tca', 'max_risk_estimate', 'max_risk_scaling', 
                'mahalanobis_distance', 'miss_distance', 'c_position_covariance_det', 
                'c_obs_used', 'number_CDMs', 'mean_risk_CDMs', 'std_risk_CDMs']]
    
    # Extracting low risk data and associated classes
    X_train = X_train[(y_train['is_non_anomalous_lr'] == 1) | (y_train['is_anomalous_hr'] == 1)]
    y_train = y_train[(y_train['is_non_anomalous_lr'] == 1) | (y_train['is_anomalous_hr'] == 1)]

    # Keeping the relevant target only
    y_train = y_train['is_anomalous_hr']
    y_train.rename('anomalous', inplace=True)
    
    return X_train, y_train


def create_test_dataset(X_test, y_test):
    """
    Create test dataset for the task by computing new features and selecting relevant data. 

    Parameters:
    - X_train (pandas DataFrame): Official testing dataset unchanged downloaded from the competition site.
    - y_train (pandas DataFrame): Official targets of the testing dataset unchanged downloaded from the competition site.

    Returns:
    - X_test_lr (pandas DataFrame): High risk test dataset
    - y_test_lr (pandas DataFrame): High risk test targets
    - X_test_hr (pandas DataFrame): Low righ test dataset
    - y_test_hr (pandas DataFrame): Low risk test targets
    """

    # Copying to avoid modifying original datasets
    X_test = X_test.copy()
    y_test = y_test.copy() 

    # Adding the new features 
    grouped_data = X_test[['event_id', 'risk']].groupby(by='event_id')

    number_cdms = grouped_data.size()
    mean_risk_cdms = grouped_data.mean()
    std_risk_cdms = grouped_data.std().fillna(0)

    new_features = pd.DataFrame({
        'number_CDMs': number_cdms,
        'mean_risk_CDMs': mean_risk_cdms['risk'],
        'std_risk_CDMs': std_risk_cdms['risk']
    })
    
    X_test = pd.merge(X_test, new_features, on='event_id')

    # Extracting the last CMD available for each event
    X_test = X_test.loc[X_test.groupby('event_id')['time_to_tca'].idxmin()]

    X_test.set_index('event_id', inplace=True, drop=True)
    y_test.set_index('event_id', inplace=True, drop=True)

    # Segregate dataset into high risk and low risk datasets
    X_test_lr = X_test[X_test['risk'] < -6]
    X_test_hr = X_test[X_test['risk'] >= -6]
    y_test_lr = y_test[X_test['risk'] < -6]
    y_test_hr = y_test[X_test['risk'] >= -6]

    X_test_lr.reset_index(drop=True, inplace=True)
    X_test_hr.reset_index(drop=True, inplace=True)
    y_test_lr.reset_index(drop=True, inplace=True)
    y_test_hr.reset_index(drop=True, inplace=True) 

    # Picking relevant features only
    X_test_lr = X_test_lr[['time_to_tca', 'max_risk_estimate', 'max_risk_scaling', 
                'mahalanobis_distance', 'miss_distance', 'c_position_covariance_det', 
                'c_obs_used', 'number_CDMs', 'mean_risk_CDMs', 'std_risk_CDMs']]
    
    y_test_lr = y_test_lr['true_risk']
    
    return X_test_lr, y_test_lr, X_test_hr, y_test_hr


def create_training_pairs(X, y, undersample_ratio=None):
    """
    Create paired dataset used for training. 

    Parameters:
    - X (numpy array): Data to paired.
    - y (numpy array): Targets of data to paired. Contains the class of the data : anomalous or non-anomalous.  

    Returns:
    - X_final (torch Tensor): Paired data.
    - y_final (torch Tensor): Targets of paired data.
    """

    # Extract anomalous and non_anomalous indexes
    anomalous_index = np.where(y == 1)[0]
    non_anomalous_index = np.where(y == 0)[0]

    # Create pairs for dissimilar data
    anomalous_pairs_i, anomalous_pairs_j = np.meshgrid(non_anomalous_index, anomalous_index, indexing='ij')
    dissimilar_pairs = (anomalous_pairs_i.flatten(), anomalous_pairs_j.flatten())

    # Create pairs for similar data
    non_anomalous_pairs_i, non_anomalous_pairs_j = np.triu_indices(len(non_anomalous_index), 1)
    
    if undersample_ratio:
        nb_draw = int(len(dissimilar_pairs[0])*(1-undersample_ratio)/undersample_ratio)
        random_undersampling_indexes = np.random.choice(np.arange(0, len(non_anomalous_pairs_i), 1), nb_draw, replace=False)
        similar_pairs = (non_anomalous_pairs_i[random_undersampling_indexes], non_anomalous_pairs_j[random_undersampling_indexes])
    else:
        similar_pairs = (non_anomalous_pairs_i, non_anomalous_pairs_j)

    # Extract dissimilar data
    X_anomalous = np.stack((X[dissimilar_pairs[0]], X[dissimilar_pairs[1]]), axis=1)
    y_anomalous = np.zeros(len(X_anomalous))

    # Extract similar data
    X_non_anomalous = np.stack((X[non_anomalous_index][similar_pairs[0]], X[non_anomalous_index][similar_pairs[1]]), axis=1)
    y_non_anomalous = np.ones(len(X_non_anomalous))

    # Concatenate similar and dissimilar data to obtained the full dataset
    X_final = np.concatenate((X_anomalous, X_non_anomalous))
    y_final = np.concatenate((y_anomalous, y_non_anomalous))

    return torch.tensor(X_final, dtype=torch.float32), torch.tensor(y_final)


def create_testing_pairs(X_test, X_train, y_train, n_samples):
    """
    Create paired dataset used for inference. 

    Parameters:
    - X_test (numpy array): Unknown data we want to infere on. 
    - X_train (numpy array): Known data used to infere.
    - y_train (numpy array): Targets of knwon data. Contains the class of the data : anomalous or non-anomalous.  

    Returns:
    - X_final (torch Tensor): Paired data.
    - y_reference (torch Tensor): Targets of paired data. Contains the class of the known data (anomalous or non-anomalous) for each paired data.
    """

    # Segregate anomalous and non-anomalous data
    X_train_non_anomalous, X_train_anomalous = torch.tensor(X_train[y_train == 0], dtype=torch.float32), torch.tensor(X_train[y_train == 1], dtype=torch.float32)
    
    # Transform dataset into a torch Tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Select random non-anomalous data according to the number of sample used
    indices = torch.randperm(X_train_non_anomalous.shape[0])[:(n_samples-X_train_anomalous.shape[0])]
    X_train_non_anomalous = X_train_non_anomalous[indices]

    # Extract anomalous and non-anomalous known data and its corresponding labels
    y_reference = torch.concat([torch.zeros(X_train_non_anomalous.shape[0]), torch.ones(X_train_anomalous.shape[0])])
    X_train_trunc = torch.concat([X_train_non_anomalous, X_train_anomalous])

    # Create pairs of data : each of test data is paired with every known data
    n, m, l = X_train_trunc.shape[0], X_test.shape[0], X_train_trunc.shape[1]
    X_test_exp = X_test.unsqueeze(1).expand(-1, n, -1).reshape(n*m, l)
    X_train_trunc_exp = X_train_trunc.unsqueeze(0).expand(m, -1, -1).reshape(n*m, l)
    y_reference = y_reference.repeat(m)
    
    return torch.stack((X_test_exp, X_train_trunc_exp), dim=1), y_reference


def preprocess_data(X_train, y_train, X_test, y_test, n_splits, undersample_ratio):
    """
    Prepare datasets by normalizing them and extracting validation folds for the training process.

    Parameters:
    - X_train (pandas DataFrame): Train dataset.
    - y_train (pandas Series): Train targets. Contains the class of the data : anomalous or non-anomalous.  
    - X_test (pandas DataFrame): Test dataset.
    - y_test (pandas Series): Test targets. Contains the risk to predict.
    - n_splits (int): Number of validation folds.
    - undersample_ratio (float between 0.0 and 0.5): Ratio to balance classes. Represents the proportion of the under-represented 
                                class after the balancing process.

    Returns:
    - X_train_preprocessed (numpy array): Train dataset normalized
    - y_train_preprocces (numpy array): Train targets normalized
    - X_test_preprocessed (numpy array): Test dataset normalized
    - y_test_preprocessed (numpy array): Test targets normalized
    - datasets (List of tuple): Contains (X_train, y_train, X_val, y_val) for each validation fold
    """

    # Initialise and fit a Quantile Transformer
    quantile_transformer = QuantileTransformer().fit(X_train)
    
    # Normalise Train and Test data with Quantile Transformer
    X_train_preprocessed = quantile_transformer.transform(X_train)
    X_test_preprocessed = quantile_transformer.transform(X_test)

    # Convert targets in numpy arrays
    y_train_preprocessed = y_train.to_numpy()
    y_test_preprocessed = y_test.to_numpy()

    # Create stratified validation folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    datasets = []
    for (train_index, val_index) in skf.split(X_train_preprocessed, y_train_preprocessed):
        
        X_train_fold = X_train_preprocessed[train_index]
        y_train_fold = y_train_preprocessed[train_index]
        X_train_fold, y_train_fold = create_training_pairs(X_train_fold, y_train_fold, undersample_ratio=undersample_ratio)

        X_val_fold = X_train_preprocessed[val_index]
        y_val_fold = y_train_preprocessed[val_index]
        X_val_fold, y_val_fold = create_training_pairs(X_val_fold, y_val_fold, undersample_ratio=undersample_ratio)

        datasets.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    return X_train_preprocessed, y_train_preprocessed, X_test_preprocessed, y_test_preprocessed, datasets


def create_dataloader(X, y, batch_size):
    """
    Create torch dataloader.

    Parameters:
    - X (torch tensor): Data 
    - y (torch tensor): Targets of data
    - batch_size (int): Batch size used for dataloaders.

    Returns:
    - dataloader (torch Dataloader): Associated dataloader.
    """

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_model(model, train_dataloader, val_dataloader, n_epochs, lr, clip_gradient_value, early_stopping_threshold, model_save_path, verbose):
    """ 
    Train a model.

    Parameters:
    - model (torch Module): Torch model to train.
    - train_dataloader (torch DataLoader): Data used to train the model.
    - val_dataloader (torch DataLoader): Data used for validation steps.
    - n_epochs (int): Number of training steps.
    - lr (float): Learning rate used to update weights.
    - clip_gradient_value (float): Value of gradient clip.
    - early_stopping_threshold (int): Number of epochs without improvment of val loss allowed. 
                                    If the threshold is reach, training process is stopped and the model 
                                    state with the lowest val loss is retrieved.
    - model_save_path (str path): Path to save the trained model.
    - verbose (bool): If some verbose is needed.

    Returns:
    - model (torch Module): Trained model
    """

    # Create Loss function and Optimizer
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Migrate model to device
    model.to(device)

    # Initialise variables
    early_stopping_counter = 0
    best_val_loss = torch.inf
    best_epoch = 0

    # Loop on epochs
    for epoch in range(n_epochs):
        model.train()
        total_train_loss_epoch = 0

        # Training step
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data[:, 0, :], data[:, 1, :])
            loss = loss_function(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_value)
            optimizer.step()
            total_train_loss_epoch += loss.item()

        # Validation step
        model.eval()
        total_val_loss_epoch = 0
        
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data[:, 0, :], data[:, 1, :])
                loss = loss_function(output, target)
                total_val_loss_epoch += loss.item()

        # Early stopping parameters
        if total_val_loss_epoch/len(val_dataloader) < best_val_loss:
            early_stopping_counter = 0
            best_val_loss = total_val_loss_epoch/len(val_dataloader)
            best_epoch = epoch
            torch.save(model, model_save_path)
            
        else: 
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_threshold:

            if verbose:
                print(f"Early stopping triggered : Stopping on epoch {best_epoch+1} with validation loss {best_val_loss}")
            break

        if verbose:
            print(f'Epoch {epoch+1}, Training Loss: {total_train_loss_epoch/len(train_dataloader)}, Validation Loss: {total_val_loss_epoch/len(val_dataloader)}')

    # Load the best state of the model
    model = torch.load(model_save_path)
    
    return model


def final_training(X_train, y_train, X_val, y_val, params_list, verbose):
    """ 
    Realises the final training of a list of models according to params_list.

    Parameters:
    - X_train (int): Paired data used to train models.
    - y_train (int): Labels of X_train : 1 for {non-anomalous, non-anomalous} couples, 0 for {anomalous, non-anomalous} couples.
    - X_val (torch tensor): Paired data used for validation steps.
    - y_val (torch tensor): Labels of X_val : 1 for {non-anomalous, non-anomalous} couples, 0 for {anomalous, non-anomalous} couples.

    Returns:
    - models (list): Stores path of trained models.
    """

    # Initialise the list
    models = []

    # Loop on the params
    for i, params in enumerate(params_list):

        # Create and save model into the list
        model = ManhattanLSTM(hidden_size=int(params["hidden_size"]), num_layers=1, input_size=1)
        model_save_path = f".\\MaLSTM_models\\model{i}.pth"
        models.append(model_save_path)

        if verbose:
            print(f'\n\n============ Final Training for configuration n°{params} ============\n')
        
        # Create Train and Test dataloaders
        train_dataloader = create_dataloader(X_train_final, y_train_final, batch_size=int(params['batch_size']))
        val_dataloader = create_dataloader(X_val_final, y_val_final, batch_size=int(params['batch_size']))
        
        # Train model
        model = train_model(model, train_dataloader, val_dataloader, n_epochs=10000, lr=params["lr"], 
                    clip_gradient_value=params['clip_gradient_value'], early_stopping_threshold=params['early_stopping_threshold'], model_save_path=model_save_path)

    return models
    

def compute_auc(model, X, y):
    """
    Compute AUC score on data X predicted with model.

    Parameters:
    - X (torch tensor): Data to predict
    - y (torch tensor): Targets of data
    - batch_size (int): Batch size used for dataloaders.

    Returns:
    - auc (float): Score - Best value is 1.0
    """

    # Migrate objects to the cpu
    model, X, y = model.to("cpu"), X.to("cpu"), y.to("cpu")
    model.eval()

    # Compute predictions
    with torch.no_grad():
        output = model(X[:, 0, :], X[:, 1, :])
        y_pred = torch.sigmoid(output).numpy()

    # Compute auc
    auc = roc_auc_score(y, y_pred)

    return auc


def compute_prediction_from_model_output(nb_pred, y_output, nb_reference, y_reference):
    """ 
    Compute the final class of data associated to y_output.

    Parameters:
    - nb_pred (int) : Number of data to predict
    - nb_reference (int) : Number of data used as reference
    - y_output (torch tensor) : Predictions made by the model for test data
    - y_reference (torch tensor) : Labels of reference data used to extract the class of the test data 

    Returns:
    - y_pred (torch tensor): Predicted classes of y_output
    """
    
    # Initialise tensors containing predicted probabilities
    anomalous_prob, non_anomalous_prob = torch.zeros(nb_pred), torch.zeros(nb_pred)
    
    # Loop on the number of data to predict
    for i in range(nb_pred):

        # Extract known anomalous data associated to the current data we want to predict
        anomalous_index = (y_reference[i*nb_reference:(i+1)*nb_reference] == 1)

        # Extract known non-anomalous data associated to the current data we want to predict
        non_anomalous_index = (y_reference[i*nb_reference:(i+1)*nb_reference] == 0)

        # Compute mean probability to be anomalous for the current data
        anomalous_prob[i] = y_output[i*nb_reference:(i+1)*nb_reference][anomalous_index].mean().item()

        # Compute mean probability to be non-anomalous for the current data
        non_anomalous_prob[i] = y_output[i*nb_reference:(i+1)*nb_reference][non_anomalous_index].mean().item()

    # Extract a class from the computed probabilities
    y_pred = torch.where(anomalous_prob >= non_anomalous_prob, 1.0, 0.0)
    
    return y_pred


def predict_hr(X):
    """ 
    Compute the prediction of a high risk data. The method is a simple LRP.

    Parameters:
    - X (pandas DataFrame) : Data to predict. Must contain a column 'risk'

    Returns:
    - y_pred (torch tensor): Predicted final risk of X
    """

    return X['risk'].values


def predict_lr(models, X_test, X_train, y_train, n_samples):

    """
    Generate risk prediction for X_test.

    Parameters:
    - models (sequence of path): List of trained models used to generate predictions
    - X_test (numpy array): Testing data we want to generate predictions on. X_test must contain only low risks event.
    - X_train (numpy array): Reference data used to predict a class for X_test. 
    - y_train (numpy array): Labels of X_train used to extract the class of X_test
    - n_samples (int): Number of data in X_train to use to predict a class for X_test

    Returns:
    - y_pred_lr (torch tensor): Predicted risk for X_test
    """

    # Initialise list of predictions
    y_preds = []

    for model in models:

        # Load model
        model = torch.load(model)
        
        # Create paired dataset to compute similarity between test and train data
        X_test_paired, y_reference = create_testing_pairs(X_test, X_train, y_train, n_samples)

        # Migrate objects to cpu
        model, X_test_paired, y_reference = model.to("cpu"), X_test_paired.to("cpu"), y_reference.to("cpu")
        model.eval()

        # Compute the probability that paired data are similar
        with torch.no_grad():
            output = model(X_test_paired[:, 0, :], X_test_paired[:, 1, :])
            y_output = torch.sigmoid(output).numpy()

        nb_pred, nb_reference = X_test.shape[0], n_samples

        # Extract predicted class by the model
        y_pred = compute_prediction_from_model_output(nb_pred, y_output, nb_reference, y_reference)
        y_preds.append(y_pred)

    # Associate predictions of all models by taking the mean and extract the corresponding risk value
    y_preds_tot = torch.stack(y_preds, dim = 1)
    y_pred_lr = torch.where(y_preds_tot.mean(dim=1)>0.5, -5.35, -6.001)

    return y_pred_lr