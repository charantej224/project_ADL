import os
import json

root_dir = '/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project'

dept_metrics = 'dept_classification/dept_metric.json'
prob_metrics = 'problem_classification/prob_metric.json'


def read_json(file_path):
    with open(file_path, 'r') as f:
        val_dict = json.load(f)
        f.close()
        return val_dict


dept_metrics = os.path.join(root_dir, dept_metrics)
prob_metrics = os.path.join(root_dir, prob_metrics)
dept_metrics = read_json(dept_metrics)
prob_metrics = read_json(prob_metrics)


def calculate_time(metrics, category_type):
    train_time = 0
    val_time = 0
    for i in range(10):
        train_time = train_time + metrics['train_time_' + str(i)]
        val_time = val_time + metrics['val_time_' + str(i)]
    print(category_type)
    print(f'train time - {train_time} ; validation time - {val_time}')
    print('total time {}'.format(train_time + val_time))


calculate_time(dept_metrics, 'dept_metrics')
calculate_time(prob_metrics, 'prob_metrics')
