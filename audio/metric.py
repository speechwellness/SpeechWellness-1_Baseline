from sklearn import metrics

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute(self, predictions, references, normalize=True, sample_weight=None):

        metric_dic = {}
        if self.num_classes == 1:
            metric_dic["rmse"] = metrics.root_mean_squared_error(y_true=references, y_pred=predictions, squared=False)
        else:
            metric_dic["accuracy"] = metrics.accuracy_score(references, predictions)
            if self.num_classes == 2:
                metric_dic["f1"] = metrics.f1_score(references, predictions)
                metric_dic["precision"] = metrics.precision_score(references, predictions)
                metric_dic["recall"] = metrics.recall_score(references, predictions)
            else:
                metric_dic["f1"] = metrics.f1_score(references, predictions, average='macro')
                metric_dic["precision"] = metrics.precision_score(references, predictions, average='macro')
                metric_dic["recall"] = metrics.recall_score(references, predictions, average='macro')

        return metric_dic
    

def metrics_compute(predictions, references, num_classes=2) -> dict:
    metric_dic = {}
    metric_dic["accuracy"] = metrics.accuracy_score(references, predictions)
    if num_classes == 2:
        metric_dic["f1"] = metrics.f1_score(references, predictions)
        metric_dic["precision"] = metrics.precision_score(references, predictions)
        metric_dic["recall"] = metrics.recall_score(references, predictions)
    else:
        metric_dic["f1"] = metrics.f1_score(references, predictions, average='macro')
        metric_dic["precision"] = metrics.precision_score(references, predictions, average='macro')
        metric_dic["recall"] = metrics.recall_score(references, predictions, average='macro')

    return metric_dic