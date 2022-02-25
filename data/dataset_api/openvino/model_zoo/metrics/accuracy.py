from ..api.metrics import Metric

def create_accuracy(top_k=1, name='accuracy'):
    return Metric.provide('accuracy', {'type': 'accuracy', 'top_k': top_k}, None, name)
