import matplotlib.pyplot as plt
import os
import json

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
department_metrics = os.path.join(root_dir, "dept_classification/dept_metric.json")
problem_metrics = os.path.join(root_dir, "problem_classification/prob_metric.json")


def plot_accuracy(values1, values2, x_label, y_label, plot_title, file_name):
    plt.plot(values1)
    plt.plot(values2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend(['training_accuracy', 'validation_accuracy'], loc='upper left')
    plt.savefig(file_name)
    plt.cla()


def get_values(input_metrics, plot_title, file_name):
    train_accuracy = []
    validation_accuracy = []
    with open(input_metrics, 'r') as f:
        metrics_dict = json.load(f)
        f.close()
    for i in range(10):
        train_accuracy.append(metrics_dict['train_accuracy_' + str(i)])
        validation_accuracy.append(metrics_dict['val_accuracy_' + str(i)])
    plot_accuracy(train_accuracy, validation_accuracy, 'Accuracy', 'Epoch', plot_title, file_name)


if __name__ == '__main__':
    get_values(department_metrics, "Accuracy for Department Prediction", "department_prediction.png")
    get_values(problem_metrics, "Accuracy for Problem Prediction", "problem_prediction.png")
