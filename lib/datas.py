def make_Sequence(time_lag,data,pred_length=5):
    """
    Generate time-delay sequence data 

    Args: 
        time_lag: Time lag for the sequence
        data: A numpy array follows [Ntime, Nmode] shape
        pred_length: Prediction length for the sequence

    Returns:
        X: Numpy array for Input 
        Y: Numpy array for Output
    """

    from tqdm import tqdm 
    import numpy as np 

    nSamples    = data.shape[0]-time_lag-pred_length
    X           = np.zeros([nSamples, time_lag,    data.shape[-1]]).astype(np.float32)
    Y           = np.zeros([nSamples, pred_length, data.shape[-1]]).astype(np.float32)
    # Fill the input and output arrays with data
    for j in tqdm(np.arange(data.shape[0]-time_lag-pred_length)):
        X[j] = data[j: j+time_lag,:]
        Y[j] = data[j+time_lag: j+time_lag+pred_length, :]

    return X, Y

def sample_series(series, n_samples, time_lag=64,train_ahead=5):
    """
    Randomly sample n_samples from a series.
    """

    import numpy as np 
    if series.shape[0] - time_lag - train_ahead < n_samples:
        raise ValueError("n_samples must be less than or equal to the length of the series")
    rng = np.random.default_rng(42)  # Set the random seed for reproducibility
    if n_samples != 0:
        indices = rng.choice(series.shape[0] - time_lag - train_ahead, n_samples,
                            replace=True, shuffle=False)
    else:
        indices = np.arange(series.shape[0] - time_lag - train_ahead)

    indices = np.sort(indices)

    # Check other dimensions of series
    if len(series.shape) == 1:
        series = series[:, np.newaxis]
        
    X = np.zeros((n_samples, time_lag) + series.shape[1:])
    Y = np.zeros((n_samples, train_ahead) + series.shape[1:])
    for i, idx in enumerate(indices):
        X[i] = series[idx:idx + time_lag]
        Y[i] = series[idx + time_lag:idx + time_lag + train_ahead]
    return X, Y

def sample_series_indices(series_length, n_samples, time_lag=64, train_ahead=5, seed=42):
    """
    Randomly generate indices to slice a series of given length into (X, Y) pairs.
    Returns the starting indices of the input sequences.
    """
    import numpy as np
    if series_length - time_lag - train_ahead < n_samples and n_samples != 0:
        raise ValueError("n_samples must be less than or equal to the length of the series")

    rng = np.random.default_rng(seed)
    if n_samples != 0:
        indices = rng.choice(series_length - time_lag - train_ahead, n_samples, replace=True, shuffle=False)
    else:
        indices = np.arange(series_length - time_lag - train_ahead)
    return np.sort(indices)



def make_dataloader(X, Y, batch_size=32, shuffle=True, distributed=False):
    """
    Create a DataLoader for the dataset.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create a TensorDataset
    dataset = TensorDataset(X, Y)
    
    # Create a DataLoader
    if distributed:
        import os
        from torch.utils.data.distributed import DistributedSampler
        world_rank = int(os.environ["RANK"])
        sampler = DistributedSampler(dataset, seed=world_rank)  # Ensure different shuffling for each process
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    

def normalize_data(data, mean, std):
    return (data - mean) / std
def denormalize_data(data, mean, std):
    return (data * std) + mean



        