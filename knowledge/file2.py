## file2.py
from file3 import transform

def process_data(data):
    transformed = [transform(x) for x in data]
    return sum(transformed)
