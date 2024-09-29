import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD
from pyod.models.suod import SUOD
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
# from joblib import dump, load
from sklearn.cluster import SpectralClustering, OPTICS
from utils import *
import clustered_data_reader
import change_point_detector
import pickle
import sys
import os
import distance


def load_data(data_path, scenario):
    cdr = clustered_data_reader.ClusteredDataReader(data_path)
    if scenario == "CA":

        # Initiating training data
        train = cdr.train_tasks
        unique_task = []
        train_changes = []
        data_index = 0
        for train_task in train:
            unique_task.append(train_task.data)
            data_index += train_task.data.shape[0]
            train_changes.append(data_index)

        train_changes.pop() # Removing last element of list that signals end of dataset
        train_data = np.vstack(unique_task)

        # Initiating testing data
        test_tasks = cdr.load_test_tasks()


        return train_data, train_changes, test_tasks
    else: 
        # Initiating training data
        train_tasks = cdr.train_tasks

        # Initiating testing data
        test_tasks = cdr.load_test_tasks()
        return train_tasks, test_tasks

def sample_rows(array, sample_percent):
    num_samples = round(array.shape[0]*sample_percent)
    row_indices = np.random.choice(array.shape[0], size=num_samples, replace=False)
    sampled_rows = array[row_indices]
    return sampled_rows


def update_model(scenario, s, clf, train_b, i = None, tasks_train=None, train_changes=None): # i, tasks_train needed for CI; train_changes needed for CA
    if scenario == "CI":
        if s == 'Naive':
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)
        elif s == 'Cumulative':
            if i > 0:
                train_tasks_dfs = []
                for k in range(0, i + 1):  # including current task
                    train_tasks_dfs.append(tasks_train[k])
                train_b = np.vstack(tuple(train_tasks_dfs))
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)
        # elif s == 'MSTE':
        #     clf.fit(train_b)
        #     MSTE.append(clf)
        elif s == 'Replay':
            train_tasks_dfs = []
            if i > 0:
                for k in range(0, i):
                    rb = sample_rows(tasks_train[k], rep_budget)
                    train_tasks_dfs.append(rb)
            if rep_compact:
                rb = sample_rows(tasks_train[i], rep_budget)
                train_tasks_dfs.append(rb)
            else:
                train_tasks_dfs.append(tasks_train[i])
            train_b = np.vstack(tuple(train_tasks_dfs))
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)

        else:
            print('Unknown strategy')
            sys.exit(1)


    elif scenario == "CA":
        if len(train_changes) == 1: # to handle instances where no changes are detected
            train_changes = [0] + train_changes
        if s == 'Naive':
            batch_lower, batch_upper = train_changes[-2], train_changes[-1]
            # print("Batch lower, upper:", batch_lower, batch_upper)
            train_b = train_b[batch_lower: batch_upper]
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)
        elif s == 'Cumulative':
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)
        # elif s == 'MSTE':
        #     clf.fit(train_b)
        #     MSTE.append(clf)
        elif s == 'Replay':
    
            if len(train_changes) > 2:
                train_tasks_dfs = []
                for i in range(len(train_changes)-1):
                    data_slice = train_b[train_changes[i]:train_changes[i+1]]
                    rb = sample_rows(data_slice, rep_budget)
                    train_tasks_dfs.append(rb)
                if rep_compact:
                    final_task = train_b[train_changes[-2]:train_changes[-1]]
                    rb = sample_rows(data_slice, rep_budget)
                    train_tasks_dfs.append(rb)
                else: 
                    final_task = train_b[train_changes[-2]:train_changes[-1]]
                    train_tasks_dfs.append(final_task)

                train_b = np.vstack(tuple(train_tasks_dfs))
            elif rep_compact:
                data_slice = train_b[train_changes[-2]:train_changes[-1]]
                rb = sample_rows(data_slice, rep_budget)
            else: 
                train_b # No operation done to the data frame
            # print(f'Train {len(train_b)}')
            clf.fit(train_b)
        else:
            print('Unknown strategy')
            sys.exit(1)
    
    else: 
        print("Unknown scenario")
        sys.exit(1)



# ****************************************************
# Evaluation on all tasks (all strategies except MSTE)
# ****************************************************

# To get essentially probability scores for the sklearn functions. 

def normalize_score(scores): 
    for index, score in enumerate(scores):
        scores[index] =  1/(1+np.exp(score))
    return scores

def evaluate_model(clf, i, tasks_eval, labels_eval, eval_standard_metrics = False):
    num_tasks = len(tasks_eval)

    for j in range(num_tasks):
        eval_b = tasks_eval[j]
        # print(f'Eval {j} - Len: {len(eval_b)}')
        eval_labels_b = labels_eval[j]
        # print(f'{sum(1 for anom in eval_labels_b if anom == 1)}/{len(eval_b)} anomalies')

        preds_raw = clf.decision_function(eval_b)
        # print("Labels", eval_labels_b)
        # print("Predictions", preds_raw)
        # ROC = roc_auc_score(eval_labels_b, preds_raw)
        # print(f'ROC: {ROC}')
        

        # if "sklearn" in inspect.getmodule(clf).__name__:
        #     preds = clf.predict(eval_b)
        #     preds = [1 if x == -1 else 0 for x in preds]
        #     print("Labels", eval_labels_b)
        #     print("Predictions", preds) 
        # else:
        #     preds = clf.predict(eval_b)
        # ROC = roc_auc_score(eval_labels_b, preds)
        
        preds = clf.predict(eval_b)
        if not str(clf).__contains__("contamination"):
            preds_raw = normalize_score(preds_raw)
            np.place(preds, preds == 1, 0)
            np.place(preds, preds == -1, 1)
        ROC = roc_auc_score(eval_labels_b, preds_raw)

        # print(preds)
        # print(eval_labels_b)
            
        mat_roc[i][j] = np.round(ROC, 2)

        if eval_standard_metrics:
            cf = confusion_matrix(eval_labels_b, preds, labels=[0, 1])
            print(cf)

            [precision_micro, recall_micro, fscore_micro, support_RF_micro] = precision_recall_fscore_support(eval_labels_b, preds, average='micro')
            micro_metrics = [precision_micro, recall_micro, fscore_micro, support_RF_micro]
            [precision_macro, recall_macro, fscore_macro, support_RF_macro] = precision_recall_fscore_support(eval_labels_b, preds, average='macro')
            macro_metrics = [precision_macro, recall_macro, fscore_macro, support_RF_macro]
            [precision_weighted, recall_weighted, fscore_weighted, support_RF_weighted] = precision_recall_fscore_support(eval_labels_b, preds, average='weighted')
            weighted_metrics = [precision_weighted, recall_weighted, fscore_weighted, support_RF_weighted]
            print(f'{precision_micro},{recall_micro},{fscore_micro}')
            print(f'{precision_macro},{recall_macro},{fscore_macro}')
            print(f'{precision_weighted},{recall_weighted},{fscore_weighted}')
            mat_f1[i][j] = np.round(fscore_weighted, 2)
            
    
    if eval_standard_metrics:
        return ROC, mat_roc, micro_metrics, macro_metrics, weighted_metrics
    else:
        return ROC, mat_roc

# *************************************************
# *************************************************
# scenario = "CI"

scenario = "CA"


detecing_points = False
eval_standard_metrics = False

drop_cols = ["anomaly"]

distance_metric = "CA Vicis Wave Hedges"
location_dict = {"energy-A": [5984, 11968, 17952, 23936, 30056, 31688, 36176, 41888, 48416, 49776, 53992], 
                 "energy-C": [5984, 11968, 17952, 23936, 30056, 42024, 48008, 53992],
                 "energy-R": [1360, 5984, 11968, 17952, 23936, 30056, 31280, 36040, 41888, 48008, 48824, 53992],
                 "nslkdd-A": [300, 8300, 8650, 9200, 9850, 10500, 10700, 11100, 11450, 11750, 11950, 12450, 17700, 18200, 18900, 21000, 22850, 23450, 26850, 27300, 27500, 28450, 28800],
                 "nslkdd-C": [200, 850, 1100, 2100, 8250, 8600, 8900, 9100, 9850, 11250, 11450, 12000, 12350, 12550, 12850, 13450, 14050, 14350, 15800, 16050, 16300, 16550, 16750, 16950, 17250, 17700, 18150, 22850, 23400, 26850, 27250, 27550, 27750, 28200, 28400, 28600],
                 "nslkdd-R": [300, 1050, 8200, 8600, 9200, 9600, 9850, 10450, 10800, 11450, 11900, 16250, 16450, 17100, 17300, 17500, 17700, 17900, 18100, 22850, 23400, 26700, 27050, 27300, 28200, 28450, 29150, 30450],
                 "unsw-A": [15050, 49000, 55650, 59500], 
                 "unsw-C": [1050, 2100, 6650, 8050, 9450, 49000, 50750, 52150, 54950, 56000],
                 "unsw-R": [1050, 2100, 3150, 4200, 5250, 6300, 7350, 8400, 9450, 10500, 11550, 12600, 13650, 14700, 49000, 50050, 55650, 66150, 67200],
                 "wind-A": [2312, 8976, 14008],
                 "wind-C": [8976, 9656, 11560],
                 "wind-R": [8976, 10336]
}

dataset_name_dict = {"energy-A": "data/energy_random_anomalies_10_concepts_10000_per_cluster.npy",
                     "energy-C": "data/energy_clustered_with_closest_assignment_10_concepts_10000_per_cluster.npy",
                     "energy-R": "data/energy_clustered_with_random_assignment_10_concepts_10000_per_cluster.npy",
                     "nslkdd-A": "data/nsl-kdd_random_anomalies_20_concepts_5000_per_cluster.npy",
                     "nslkdd-C": "data/nsl-kdd_clustered_with_closest_assignment_20_concepts_5000_per_cluster.npy",
                     "nslkdd-R": "data/nsl-kdd_clustered_with_random_assignment_20_concepts_5000_per_cluster.npy",
                     "unsw-A": "data/unsw_random_anomalies_10_concepts_30000_per_cluster.npy", 
                     "unsw-C": "data/unsw_clustered_with_closest_assignment_10_concepts_30000_per_cluster.npy",
                     "unsw-R": "data/unsw_clustered_with_random_assignment_10_concepts_30000_per_cluster.npy",
                     "wind-A": "data/wind_random_anomalies_5_concepts_15000_per_cluster.npy",
                     "wind-C": "data/wind_clustered_with_closest_assignment_5_concepts_15000_per_cluster.npy",
                     "wind-R": "data/wind_clustered_with_random_assignment_5_concepts_15000_per_cluster.npy"
}

batch_size_dict = {"energy-A": 136,
                     "energy-C": 136,
                     "energy-R": 136,
                     "nslkdd-A": 50,
                     "nslkdd-C": 50,
                     "nslkdd-R": 50,
                     "unsw-A": 1050, 
                     "unsw-C": 1050,
                     "unsw-R": 1050,
                     "wind-A": 136,
                     "wind-C": 136,
                     "wind-R": 136
                     }



models = [LocalOutlierFactor(n_neighbors=10, novelty=True),
          IsolationForest(random_state=0),
          ABOD(contamination=0.01),
          HBOS(contamination=0.01),
          COPOD(contamination=0.01),
          AutoEncoder(hidden_neurons=[20, 10, 5, 10, 20], epochs=20, contamination=0.01, batch_size=64, dropout_rate=0, l2_regularizer=0, validation_size=0, preprocessing=False, verbose=1),
          OneClassSVM(gamma='auto', kernel='rbf', cache_size = 1000),
          OneClassSVM(gamma='auto', kernel='sigmoid', cache_size = 1000),
          ]



#strategy = 'Replay Graph'
rep_budget = .2
rep_compact = True      # Use a sample from last task instead of full data
const_ratio_replay = False
min_samples_cluster = False
skip_noise = False

strategies = ["Naive", "Replay", "Cumulative"]
# strategies = ["Naive"]


params = {
    "Naive": "",
    "Replay": f'budget={rep_budget}_compact_{rep_compact}',
    "Cumulative": ""
}


tasks_train = []
tasks_eval = []
labels_eval = []

MSTE = []

# *************************************************
# *************************************************
for dataset_name in list(location_dict.keys()):  
    
    tasks_train = []
    tasks_eval = []
    labels_eval = []

    MSTE = [] 

    task_index = 0 
    batch_size = batch_size_dict[dataset_name]

    data_path = dataset_name_dict[dataset_name]
    # Load data
    if scenario == "CI":
        train_data, test_tasks = load_data(data_path, scenario)
    else: 
        train_data, train_changes_true, test_tasks = load_data(data_path, scenario)
    num_tasks = len(test_tasks)

    for i in range(num_tasks):
        if scenario == "CI":
            train_b = np.array(train_data[i].data)
            # train_b.drop(columns=drop_cols, inplace=True)   # Remove class attribute from training sets
            tasks_train.append(train_b)
        # eval_b = pd.read_csv(f'{dataset_name}_{i}_eval.csv')
        # eval_labels = eval_b["anomaly"]
        # eval_b.drop(columns=drop_cols, inplace=True)
        # tasks_eval.append(eval_b)
        # labels_eval.append(eval_labels)
        tasks_eval.append(test_tasks[i].data)
        labels_eval.append(test_tasks[i].labels)

    # Run scenario (train and predict)
    for s in strategies:
        print(f'Strategy: {s}')
        logger = []
        logger.append('strategy,params,model,LROC,BWT,FWT')

        for m in models:
            clf = m

            mat_f1 = np.zeros((num_tasks, num_tasks))
            mat_roc = np.zeros((num_tasks, num_tasks))
            print(f'__________________\n{m}\n__________________\n')

            # Concept Agnostic: Unique stream + change point detection to detect concept changes
            if scenario == 'CA':
                if detecing_points:
                    thresholds = [1.17]
                    metric = distance.kulczynski
                    locations_by_threshold = []
                    for threshold in thresholds:
                        detector = change_point_detector.ModularDetector(batch_size = batch_size, threshold=threshold, max_dist_size=0, dist_metric = metric, new_dist_buffer_size=batch_size*3)
                        locations = detector.detect(train_data)
                        locations_by_threshold.append(locations)
                else: 
                    locations_by_threshold = [list(location_dict[dataset_name])]
                # Train initial model
                initial_locations = [0, batch_size*3]
                train_data_initial = train_data[:batch_size*3]
                update_model(scenario, s, clf, train_data_initial, train_changes = initial_locations)
                task_index = 0 
                for locations in locations_by_threshold:
                    locations.append(len(train_data)) # Adding end of data frame to location list
                    train_changes = train_changes_true + [len(train_data)] # Adding end of data frame to true change list
                    print("True changes", train_changes)
                    print("Detected changes", locations)
                    
                    for location in locations:
                        location_index = locations.index(location)
                        # print("Processing detected change at location", location)
                        # Overshot the true change point
                        if location > train_changes[task_index]:
                            while location > train_changes[task_index]: # Testing the model for each task missed by the change point detector

                                # Evaluating model before updating on new data
                                # print("Evaluating model")
                                if eval_standard_metrics:
                                    ROC, mat_roc, micro_metrics, macro_metrics, weighted_metrics = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)                            
                                else: 
                                    ROC, mat_roc = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)

                                # Moving to next task
                                task_index += 1

                            # Updating model using new change point
                            # print("Updating model")
                            train_data_short = train_data[:locations[location_index]]
                            locations_short = [0] + locations[:location_index+1]
                            update_model(scenario, s, clf, train_data_short, train_changes = locations_short)

                        # Hit the true change point
                        elif location == train_changes[task_index]:
                            train_data_short = train_data[:locations[location_index]]
                            locations_short = [0] + locations[:location_index+1]
                            update_model(scenario, s, clf, train_data_short, train_changes = locations_short)
                            # print("Evaluating model")
                            if eval_standard_metrics:
                                ROC, mat_roc, micro_metrics, macro_metrics, weighted_metrics = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)                            
                            else: 
                                ROC, mat_roc = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)
                            task_index += 1

                        # Undershot the true change point
                        else:
                            # print("Updating model")
                            train_data_short = train_data[:locations[location_index]]
                            locations_short = [0] + locations[:location_index+1]
                            # print(locations_short)
                            update_model(scenario, s, clf, train_data_short, train_changes = locations_short)


                    # Test model on final task if missing tasks
                    if task_index < len(train_changes):
                        # print("Updating model")
                        update_model(scenario, s, clf, train_data, train_changes = locations)
                        while task_index < len(train_changes):
                            # print("Evaluating Model")
                            if eval_standard_metrics:
                                ROC, mat_roc, micro_metrics, macro_metrics, weighted_metrics = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)
                            else: 
                                ROC, mat_roc = evaluate_model(clf, task_index, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)
                            task_index += 1

                l_roc = lifelong_roc(mat_roc)
                bwt = backward_transfer(mat_roc)
                fwt = forward_transfer(mat_roc)

                print("Lifelong ROC", l_roc)
                print("Backward transfer", bwt)
                print("Forward transfer", fwt)





            else:
                for i in range(num_tasks):
                    # print(f'Current task: {i}')

                     # CI : Concept incremental -  proceed as usual: for loop through tasks : no need for CPD, train using a single batch per task
                    train_b = tasks_train[i]
                    update_model(scenario, s, clf, train_b, i, tasks_train)

                    if eval_standard_metrics:
                        ROC, mat_roc, micro_metrics, macro_metrics, weighted_metrics = evaluate_model(clf, i, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)
                    else: 
                        ROC, mat_roc = evaluate_model(clf, i, tasks_eval, labels_eval, eval_standard_metrics = eval_standard_metrics)
                l_roc = lifelong_roc(mat_roc)
                bwt = backward_transfer(mat_roc)
                fwt = forward_transfer(mat_roc)

                print("Lifelong ROC", l_roc)
                print("Backward transfer", bwt)
                print("Forward transfer", fwt)


                # matrices_f1["naive"][str(m)] = mat_f1
                # matrices_roc["naive"][str(m)] = mat_roc

            # print(mat_roc)


            heatmap(mat_roc, distance_metric, dataset_name, s, params[s], str(m), "f")
            heatmap(mat_roc, distance_metric, dataset_name, s, params[s], str(m), "b")
            heatmap(mat_roc, distance_metric, dataset_name, s, params[s], str(m), "all")



            if "AutoEncoder" in str(m):
                m = "AutoEncoder"

            # print('strategy,params,model,LROC,BWT,FWT')
            # print(f'{s},{params[s]},{m},{l_roc},{bwt},{fwt}')
            logger.append(f'{s},{params[s]},{str(m).replace(",","").replace(" ","_")},{l_roc},{bwt},{fwt}')

            with open(f'logs/{distance_metric}/{dataset_name}_{s}_{params[s]}_{m}_rocmat.pkl', "wb") as fp:

                pickle.dump(mat_roc, fp)

        # print(logger)

        if os.path.isfile(f'logs/{distance_metric}/{dataset_name}_{s}_{params[s]}_metrics.csv'):
            with(open(f'logs/{distance_metric}/{dataset_name}_{s}_{params[s]}_metrics.csv', "a")) as f:
                for line in logger:
                    f.write(f'{line}\n')
            f.close()
        else:
            np.savetxt(f'logs/{distance_metric}/{dataset_name}_{s}_{params[s]}_metrics.csv', logger, delimiter=',', fmt='%s')

        # Load saved ROC matrix and print
        with open(f'logs/{distance_metric}/{dataset_name}_{s}_{params[s]}_{m}_rocmat.pkl', "rb") as fp:
            b = pickle.load(fp)
            print(b)




