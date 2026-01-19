import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import argparse

# Defining constants for training
sequence_length = 42

ALL_FEATURES = [
    "n_flows", "n_packets", "n_bytes", "n_dest_asn", "n_dest_ports", "n_dest_ip",
    "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes", "dir_ratio_packets", "dir_ratio_bytes",
    "avg_duration", "avg_ttl"
]

SELECTED_FEATURES = [
    "tcp_udp_ratio_bytes",  "dir_ratio_bytes", "avg_duration", "avg_ttl"
]

SELECTED_FEATURES_STR = ",".join(SELECTED_FEATURES)

# Get column indices of selected features
SELECTED_INDICES = [ALL_FEATURES.index(feat) for feat in SELECTED_FEATURES]

input_size = len(SELECTED_FEATURES)

num_classes = 3
epochs = 100

# Parsing input, because I ran the script multiple times on metacentrum with diferent hyperparameters
parser = argparse.ArgumentParser(description="Set parameters based on command-line arguments")
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--preprocessing', type=str, choices=['log', 'z_score'], required=True, help='Type of preprocessing applied to the data')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and validation')
parser.add_argument('--network_type', choices=['RNN', 'LSTM', 'GRU'], default='RNN', help='Type of network: RNN, LSTM, or GRU')
parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer to use')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the network')
parser.add_argument('--dropout', type=float, default=0,help='Dropout')
parser.add_argument('--scratch_dir', type=str, required=True, help='Path to all data in scratch dir')

args = parser.parse_args()

# Tuned hyperparameters
hidden_size = args.hidden_size
network_type = args.network_type
opt = args.opt
lr = args.lr
num_layers = args.num_layers
dropout = args.dropout
batch_size = args.batch_size

# Set data paths based on preprocessing method
TRAIN_DATA_PATH = args.scratch_dir
VAL_DATA_PATH = args.scratch_dir


# Because of its size, training dataset is divided into 8 files and stored as binary data
train_npy_files = [f'{TRAIN_DATA_PATH}/group_1.npy', f'{TRAIN_DATA_PATH}/group_2.npy',f'{TRAIN_DATA_PATH}/group_3.npy', f'{TRAIN_DATA_PATH}/group_4.npy']


# Validation dataset is way smaller than training one, so I stored it in just 1 file
val_npy_file = f'{VAL_DATA_PATH}/val_group_1.npy'

# Mapping data labels to numbers, because pytorch cant work with strings as labels for data (or at least i did not find how)
mapping = {
    "end-device": 0,
    "server": 1,
    "net-device": 2
}



class LazyTimeSeriesDataset(IterableDataset):
    """
    Dataset class which loads each file of training dataset separately, so I dont have to load whole dataset into memory at once and gc can free memory of files that are not currently used.
    """
    def __init__(self, npy_files, batch_size=1024, shuffle=True):
        self.npy_files = npy_files
        self.batch_size = batch_size
        self.shuffle = shuffle

    def parse_file(self, file_path):
        d = np.load(file_path, allow_pickle=True).item()
        data = d['data']
        labels = d['annotations']
        labels = np.vectorize(mapping.get)(labels).astype(np.int32)

	# Shuffling the data in each file so I get different batches in each epoch
        if self.shuffle:
            indices = np.random.permutation(len(data))
            data = data[indices]
            labels = labels[indices]

        for i in range(0, len(data), self.batch_size):
            subset = data[i:i+self.batch_size]  # shape: (B, T, F)
            batch_data = subset[:, :, SELECTED_INDICES]
            batch_data = torch.tensor(batch_data, dtype=torch.float32)

            #batch_data = torch.tensor(data[i:i+self.batch_size], dtype=torch.float32)
            batch_labels = torch.tensor(labels[i:i+self.batch_size], dtype=torch.long)
            yield batch_data, batch_labels


    def __iter__(self):
        for file_path in self.npy_files:
            yield from self.parse_file(file_path)



class ValDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path):
        loaded = np.load(npy_path, allow_pickle=True).item()
        data = np.array(loaded['data'])  # Shape: (N, T, F)
        self.data = data[:, :, SELECTED_INDICES]  # Vectorized feature selection
        self.annotations = np.vectorize(mapping.get)(loaded['annotations']).astype(np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.annotations[idx], dtype=torch.long)
        return x, y



class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout = dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # because of batch_first=True, x is in format of [batch_size, seq_len, input_size]
        output, h_n = self.gru(x)
        out = self.classifier(h_n[-1])  # Using hidden state of last layer (block) as input to a perceptron layer for predictions
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout = dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # because of batch_first=True, x is in format of [batch_size, seq_len, input_size]
        output, (h_n, c_n) = self.lstm(x)
	# using hidden state of the last block as input to a perceptron layer for prediction
        out = self.classifier(h_n[-1])
        return out



class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout = dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #  because of batch_first=True, x is in format of [batch_size, seq_len, input_size]
        output, h_n = self.rnn(x)  # h_n: [num_layers, batch_size, hidden_size]
        out = self.classifier(h_n[-1])  # Use the hidden state from the last layer
        return out

def log_print(message, log_file):
    print(message)  # still prints to console
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

log_file = f"{args.scratch_dir}/{network_type},{hidden_size},{opt},{lr},{num_layers},{dropout},{batch_size},{args.preprocessing},{SELECTED_FEATURES_STR}.txt"
log_print(f"type: {network_type}, hidden_size: {hidden_size}, opt: {opt}, lr: {lr}, num_layers: {num_layers}, dropout: {dropout}, batch_size: {batch_size}, preprocessing: {args.preprocessing}, selected_features: {SELECTED_FEATURES_STR}", log_file)

train_dataset = LazyTimeSeriesDataset(train_npy_files, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, pin_memory=True)

val_dataset = ValDataset(val_npy_file)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Checking if GPU is available for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"device: {device}", log_file)


if network_type == "GRU":
    model = GRUClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,num_layers=num_layers, dropout=dropout)
elif network_type == "LSTM":
    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,num_layers=num_layers, dropout=dropout)
elif network_type == "RNN":
    model = RNNClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,num_layers=num_layers, dropout=dropout)
    
model.to(device)

criterion = nn.CrossEntropyLoss()

# Optimizer adamw was in the end never used
if opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif opt == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=lr)


best_val_f1 = 0.0
best_model_state = None

for epoch in range(epochs):
    # Training loop
    model.train()
    total_train_loss = 0
    train_preds, train_targets = [], []

    log_print(f"\nEpoch {epoch+1}", log_file)

    for batch_data, batch_labels in train_dataloader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        # Removing first dimension of size 1
        batch_data = batch_data.squeeze(0)
        batch_labels = batch_labels.squeeze(0)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        # Computing the gradient
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
	
    	# Storing predictions for accuracy and F1 score calculations
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(batch_labels.cpu().numpy())

    train_accuracy = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='macro')

    # Validation loop
    model.eval()
    total_val_loss = 0
    val_preds, val_targets = [], []

    # For validation loop I dont need to calculate gradient
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_data, val_labels = val_data.to(device), val_labels.to(device)
            
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_labels)

            total_val_loss += val_loss.item()

	    # Storing predictions for each batch
            val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
            val_targets.extend(val_labels.cpu().numpy())

    val_accuracy = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    
    # Storing the model in case I got new best validation F1-macro score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        torch.save(best_model_state, f'{args.scratch_dir}/{network_type},{hidden_size},{opt},{lr},{num_layers},{dropout},{batch_size},{args.preprocessing},{SELECTED_FEATURES_STR}.pth')


    log_print(f"Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}", log_file)
    log_print(f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}", log_file)
    log_print(f"Train F1 (macro): {train_f1:.4f} | Val F1 (macro): {val_f1:.4f}", log_file)
