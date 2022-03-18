import json
import random

# sample preprocess_handler (to be implemented by customer)
# This is a trivial example, where we simply generate random values
# But customers can read the data from inference_record and trasnform it into 
# a flattened json structure
def preprocess_handler(inference_record):
    event_data = inference_record.event_data
    input_data = {}
    output_data = {}

    input_data['feature0'] = random.randint(1, 3)
    #random.uniform 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    input_data['feature1'] = random.uniform(0, 1.6)
    input_data['feature2'] = random.uniform(0, 1.6)

    output_data['prediction0'] = random.uniform(1, 30)
    
    return {**input_data, **output_data}