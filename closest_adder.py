
import os
import numpy as np 
# from main_app import load_csv
import csv
file_path = os.path.join(os.getcwd(), 'lookuptable.csv')


def new_load_csv(file_path,input_vec_length):
    result = []
    try:
        print(input_vec_length)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            lines = list(reader)

        if len(lines) < 2:
            print('CSV file is empty or has no data rows')
            return []

        for i, line in enumerate(lines[1:], start=1):
            if len(line) == 9:
                data = list(map(int, line[:]))
                if not all(elem == 0 for elem in data[input_vec_length:-1]):
                        continue
                # Reverse the list of numbers < input_vec_length
                # print(data)
                filtered_data = [data[i] for i in range(len(data)) if i < input_vec_length]
                # filtered_data = [num for num in filtered_data if num != 0][::-1]
                cost = int(line[-1])  # Assume the last column is the cost
                result.append({'data': filtered_data, 'cost': cost})
    except Exception as e:
        print(f'Error reading CSV: {e}')
    return result



def hamming_distance(arr1, arr2):
    """
    Calculate the Hamming distance between two vectors.
    Hamming distance is the number of positions at which the elements are different.
    """
    return np.sum(np.array(arr1) != np.array(arr2))

def transform_input_vector(input_vector):
    """
    Transform the input vector by replacing all 0's with 1's.
    """
    return [1 if x == 0 else x for x in input_vector]

def is_valid_candidate(input_vector, candidate):
    """
    Check if the candidate vector is a valid match:
    - The length of the candidate vector should be greater than or equal to the input vector.
    - Each value in the candidate vector should be greater than or equal to the corresponding value in the input vector.
    """
    if len(candidate) < len(input_vector):
        return False
    for i in range(len(input_vector)):
        if candidate[i] < input_vector[i]:
            return False
    return True

def find_closest_element_by_transformed_vector(input_vector, data):
    """
    Find the element in the dataset closest to the transformed input vector by adding 1's wherever there are 0's
    in the input vector, prioritizing the minimum cost.

    :param input_vector: List representing the input vector.
    :param data: List of dictionaries, each containing a row of data with cost and values.
    :return: The row with the closest match (minimum cost based on transformed input).
    """
    transformed_input_vector = transform_input_vector(input_vector)
    closest_row = None
    min_cost = float('inf')
    min_distance = float('inf')
    # print(data)
    # print(transform_input_vector)
    # print("Input Vector",input_vector)
    for row in data:
        # print(row)

        if not is_valid_candidate(input_vector, row['data']):
            continue
        # Get the cost of the current row
        cost = row['cost']
        # print(row['data'])
        # Calculate the Hamming distance between the transformed input vector and the current row
        distance = hamming_distance(input_vector, row['data'])
        # if distance==0:
            # print(row['data'])
        
        # If the current cost is smaller or if the cost is the same but the distance is smaller
        if distance < min_distance or (distance == min_distance and cost < min_cost):
            closest_row = row
            min_cost = cost
            min_distance = distance
            # print(min_cost)
            # print(row['data'])
    
    return closest_row

# input_vector = [2,1,1]

# result=load_csv(file_path)

# closest_roww=find_closest_element_by_transformed_vector(input_vector,result)
# print(closest_roww)
