import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import src.labels as labels
import src.target_group_iter as TGI

reverse_train_labels = dict((v,k) for k,v in labels.label_names.items())


def find_most_frequent(train_labels):
    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = labels.label_names[int(num)]
            row.loc[name] = 1
        return row

    for key in labels.label_names.keys():
        train_labels[labels.label_names[key]] = 0

    train_labels = train_labels.apply(fill_targets, 1)
    print(train_labels.head())

    target_counts = train_labels.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
    sns.set()
    plt.figure(figsize=(15, 15))
    sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
    plt.show()
    return train_labels


def find_most_common_num_target(train_labels):
    train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"], axis=1).sum(axis=1)
    count_perc = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
    plt.figure(figsize=(20, 5))
    sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
    plt.xlabel("Number of targets per image")
    plt.ylabel("% of data")
    plt.show()
    return train_labels


def display_targets_correlation(train_labels):
    plt.figure(figsize=(15, 15))
    sns.heatmap(train_labels[train_labels.number_of_targets > 1].drop(
        ["Id", "Target", "number_of_targets"], axis=1
    ).corr(), cmap="RdYlBu", vmin=-1, vmax=1)
    plt.show()


############General data overview############
train_labels = pd.read_csv("../data/train.csv")
print(train_labels.head())
print(train_labels.shape[0])
#############################################

train_labels = find_most_frequent(train_labels)
train_labels = find_most_common_num_target(train_labels)
#display_targets_correlation(train_labels)


####################Find targets grouping############################
def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts


####################Find corelation############################
def find_corelation(name, color="Blues"):
    lyso_endo_counts = find_counts(name, train_labels)

    plt.figure(figsize=(10,3))
    sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette=color)
    plt.show()


#find_corelation("Lysosomes")


your_choice = ["Lysosomes", "Endosomes"]
your_batch_size = 3

target_it = TGI.TargetGroupIterator(reverse_train_labels, train_labels, your_choice, your_batch_size, "../data/train/")
target_it.find_matching_data_entries()
iterator = target_it.get_loader()

file_ids, images = next(iterator)

fig, ax = plt.subplots(len(file_ids),4,figsize=(20,5*len(file_ids)))
if ax.shape == (4,):
    ax = ax.reshape(1,-1)
for n in range(len(file_ids)):
    TGI.make_image_row(images[n], ax[n], target_it.make_title(file_ids[n], labels.label_names))