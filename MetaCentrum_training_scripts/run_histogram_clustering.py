import os
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import numpy as np
import math
from tslearn.clustering import TimeSeriesKMeans
from sktime.clustering.k_means import TimeSeriesKMeans as skTimeSeriesKMeans
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


RANDOM_STATE = 21


def parse_arguments() -> argparse.Namespace:
    """
    The function used to parse command line arguments and return the parsed
    arguments.
    
    Return:
    The function returns the parsed command-line arguments as an
    'argparse.Namespace' object.
    """
    parser = argparse.ArgumentParser(
        description="""

    Usage:""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--distance_metric",
        help="",
        type=str,
        metavar="STRING",
    )
    parser.add_argument(
        "--n_clust",
        help="",
        type=str,
        metavar="NUMBER",
    )
    parser.add_argument("--undersample", help="",required=True, choices=["True", "False"])
    parser.add_argument('--scratch_dir', type=str, required=True, help='Path to all data in scratch dir')
    parser.add_argument('--preprocessing', type=str, choices=['log', 'z_score'], required=True, help='Type of preprocessing applied to the data')
    return parser.parse_args()



class data_with_annotations:
    """
    Class for storing datapoints together with their annotations for
    easier manipulation with them later in code.
    """
    def __init__(self, data, annotations):
        self.data = data
        self.annotations = annotations
        

    

class TsKmeansClassifier:
    """
    Class for predicting and evaluation of clustering models ability to classify data.
    """
    def __init__(self, model, Xtrain, Ytrain):
        """
        Initializaton of the TsKmeansClassifier instance. Firstly, the training dataset is divided into clusters using the initialized model.
        Then each cluster is labeled by the annotation of the class, whose data points are the majority in a cluster. If there is a cluster with 0 
        data points in it, it is deleted.

        Params:
        model: Initialized clustering model with same user interface as 'tslearn.clustering.TimeSeriesKMeans'.
        Xtrain: List of training data in form of 'pandas.DataFrame'.
        Ytrain: List of true labels of training data.
        """
        self.model = model
        self.clusters = None
        self.cluster_annotations = []
        
        train_labels = model.fit_predict(Xtrain)
        
        #Getting numbers of occurances of each class in each cluster
        self.clusters = [{} for i in range(self.model.n_clusters)]
        for i in range(0, len(train_labels)):
            real_annotation = Ytrain[i] 
            increased_value = self.clusters[train_labels[i]].get(real_annotation, 0) + 1
            self.clusters[train_labels[i]][real_annotation] = increased_value
        for cluster in self.clusters:
            self.cluster_annotations.append(self.label_cluster(cluster))
        
        #Deletion of clusters with 0 data points
        indexes_of_zeros = [index for index, value in enumerate(self.cluster_annotations) if value == 0]
        indexes_of_zeros.reverse()
        for index_to_delete in indexes_of_zeros:
            model.cluster_centers_ = np.delete(model.cluster_centers_, index_to_delete, axis = 0)
            del self.clusters[index_to_delete]
            del self.cluster_annotations[index_to_delete]

    def to_pickle(self, file):
        """
        Method that saves the classifier as a file in Python pickle format.

        Params:
        file: name of file to store the Python pickle format of classifier.
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_pickle(cls, file):
        """
        Method that loads the classifier from a file in Python pickle format.

        Params:
        file: name of file from which is the classifier loaded.
        """
        with open(file, "rb") as f:
            return pickle.load(f)
        
    def label_cluster(self, cluster) -> any:
        """
        Method that labels the cluster with the annotation of majority class.
        
        Params:
        cluster: Dictionary containing occurances of each class in the cluster.
        
        Return:
        The method returns final label for a cluster of type 'Any', the type depends on the type of training data annotation passed into a constructor.
        """
        final_cluster_label = max(cluster, key=cluster.get,default = 0)
        return final_cluster_label

    def predict_annotations(self, X) -> list[any]:
        """
        Method for prediction of annotations for the input data.

        Params:
        X: List of data in form of 'pandas.DataFrame'.

        Return:
        The method returns list of annotations for the input data.
        """
        predicted_annotations = []
        predicted_labels = self.model.predict(X)
        for i in range(0,len(predicted_labels)):
            predicted_annotations.append(self.cluster_annotations[predicted_labels[i]])
        return predicted_annotations
    

    def evaluate_model(self, X, annotations) -> tuple[float, pd.DataFrame, float, float]:
        """
        Method for evaluating models ability to classify data

        Params:
        X: List of validation data in form of 'pandas.DataFrame'.
        annotations: List of real annotations for the validation data.

        Return:
        The function returns evaluation metrics accuracy, confusion matrix DataFrame, F1 score, and recall score in form of a Tuple.
        """
        real_annotations = pd.Series(annotations, name="Real annotations")
        predicted_annotations = pd.Series(self.predict_annotations(X), name="Predicted annotations")
        confusion_matrix_DataFrame = pd.crosstab(real_annotations, predicted_annotations)
        f1_macro = f1_score(annotations, predicted_annotations, average="macro")
        macro_recall = recall_score(annotations, predicted_annotations, average="macro")
        accuracy = accuracy_score(real_annotations, predicted_annotations)
        return accuracy, confusion_matrix_DataFrame, f1_macro, macro_recall



def get_histogram_values(arr):
    bin_edges = np.arange(0, 1.05, 0.05)  # Bin edges from 0 to 1 in steps of 0.05

    n_columns = arr.shape[1]
    hist_array = np.empty((len(bin_edges) - 1, n_columns), dtype=int)

    for i in range(n_columns):
        hist, _ = np.histogram(arr[:, i], bins=bin_edges, density=False)
        hist_array[:, i] = hist

    return hist_array  # shape: (number of bins, number of columns)


def undersample(data, annotations, percentage_to_delete, undersampled_class, random_state):
    """
    Fast undersampling function using NumPy for performance.
    """
    annotations = np.array(annotations)
    data = np.array(data)

    if undersampled_class not in annotations:
        print("Annotations do not contain undersampled class")
        return data, annotations  # Return original

    # Get indices of the class to undersample
    class_indices = np.where(annotations == undersampled_class)[0]

    # How many to delete
    n_to_delete = int(len(class_indices) * (percentage_to_delete / 100.0))

    # Randomly select indices to delete
    rng = np.random.default_rng(seed=random_state)
    delete_indices = rng.choice(class_indices, size=n_to_delete, replace=False)

    # Create mask to keep all other indices
    mask = np.ones(len(annotations), dtype=bool)
    mask[delete_indices] = False

    # Apply mask
    data = data[mask]
    annotations = annotations[mask]

    return data, annotations


def main():
    arg = parse_arguments()
    #Define constants
    N_CLUST = int(arg.n_clust)
    METRIC = arg.distance_metric
    UNDERSAMPLE = arg.undersample

    TRAIN_DATA_PATH = arg.scratch_dir
    VAL_DATA_PATH = arg.scratch_dir

    train_npy_files = [
    f'{TRAIN_DATA_PATH}/group_1.npy',
    f'{TRAIN_DATA_PATH}/group_2.npy',
    f'{TRAIN_DATA_PATH}/group_3.npy',
    f'{TRAIN_DATA_PATH}/group_4.npy'
    ]

    val_npy_file = f'{VAL_DATA_PATH}/val_group_1.npy'



    ALL_FEATURES = [
    "n_flows", "n_packets", "n_bytes", "n_dest_asn", "n_dest_ports", "n_dest_ip",
    "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes", "dir_ratio_packets", "dir_ratio_bytes",
    "avg_duration", "avg_ttl"
    ]   


    SELECTED_FEATURES = [
        "n_flows", "n_packets", "n_bytes"
    ]


    SELECTED_FEATURES_STR = ",".join(SELECTED_FEATURES)


    OUTPUT_CSV_NAME = f"{SELECTED_FEATURES_STR}-{N_CLUST}-{METRIC}--{UNDERSAMPLE}--{arg.preprocessing}"

    # Get column indices of selected features
    SELECTED_INDICES = [ALL_FEATURES.index(feat) for feat in SELECTED_FEATURES]

    # Initialize empty lists for later concatenation
    all_train_data = []
    all_train_annotations = []

    # Load each file and stack the data
    for file in train_npy_files:
        d = np.load(file, allow_pickle=True).item()
        all_train_data.append(d['data'])
        all_train_annotations.append(d['annotations'])

    # Concatenate arrays
    train_data_array = np.concatenate(all_train_data, axis=0)
    train_annotations_array = np.concatenate(all_train_annotations, axis=0)

    # Create dataset object
    Train = data_with_annotations(data=train_data_array, annotations=train_annotations_array)

    # Do the same for validation data
    v = np.load(val_npy_file, allow_pickle=True).item()
    val_annotations = v['annotations']
    val_data = v['data']
    Val = data_with_annotations(data=val_data, annotations=val_annotations)



    if UNDERSAMPLE == "True":
        Train.data, Train.annotations = undersample(Train.data, Train.annotations, 60, "end-device", RANDOM_STATE)


    N = len(Train.data)
    n_bins = 20  # From bin_edges
    n_features = Train.data[0].shape[1]

    hist_data = np.empty((N, n_bins, n_features))

    for i in range(N):
        hist_data[i] = get_histogram_values(Train.data[i])

    Train.data = hist_data


    # Suppose you have N samples
    N = len(Val.data)
    n_bins = 20  # From bin_edges
    n_features = Val.data[0].shape[1]

    # Preallocate new array for histogram results
    hist_data = np.empty((N, n_bins, n_features))

    # Get histogram values and replace original data with histogram data
    for i in range(N):
        hist_data[i] = get_histogram_values(Val.data[i])

    Val.data = hist_data

    Val.data = Val.data[:, :, SELECTED_INDICES]
    Train.data = Train.data[:,:,SELECTED_INDICES]
  
    if (METRIC == "euclidean"):
        model = TimeSeriesKMeans(
            n_clusters=N_CLUST,
            metric=METRIC,
            max_iter=30,
            n_jobs=-1,
            verbose=0,
            random_state=RANDOM_STATE,
            )
    else:
        model = skTimeSeriesKMeans(
            n_clusters=N_CLUST,
            n_init=1,
            metric=METRIC,
            max_iter=30,
            verbose=False,
            random_state=RANDOM_STATE,
            init_algorithm="kmeans++",
            averaging_method="mean",
            )
            
    clf = TsKmeansClassifier(model, Train.data, Train.annotations)
    model_accuracy, confusion_matrix, f1_macro, macro_recall = clf.evaluate_model(Val.data, Val.annotations)
    print(f"{N_CLUST};{METRIC},{SELECTED_FEATURES_STR},{UNDERSAMPLE},{arg.preprocessing},{model_accuracy},{f1_macro},{macro_recall}\n")
    clf.to_pickle(f"{arg.scratch_dir}/hist_{N_CLUST};{METRIC},{SELECTED_FEATURES_STR},{UNDERSAMPLE},{arg.preprocessing}.pkl")
    with open(
        f"{arg.scratch_dir}/{OUTPUT_CSV_NAME}.txt",
        "a",
    ) as f:
        f.write(
            f"{N_CLUST};{METRIC},{SELECTED_FEATURES_STR},{UNDERSAMPLE},{arg.preprocessing},{model_accuracy},{f1_macro},{macro_recall}\n"
        )


if __name__ == "__main__":
    main()
