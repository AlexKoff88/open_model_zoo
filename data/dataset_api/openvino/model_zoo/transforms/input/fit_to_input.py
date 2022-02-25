from ...api.input_feeder import create_input_feeder

def fit_to_input(model_inputs):
    return create_input_feeder(model_inputs)