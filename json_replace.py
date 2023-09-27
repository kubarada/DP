import json

# Load the JSON data from a file (replace 'input.json' with your JSON file)
with open('data/valid_annotations.json', 'r') as json_file:
    data = json.load(json_file)

# Function to recursively replace "test/" with ""
def replace_test(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_test(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = replace_test(item)
    elif isinstance(obj, str):
        obj = obj.replace("test/", "")
    return obj

# Replace "test/" with "" recursively in the JSON data
modified_data = replace_test(data)

# Save the modified JSON data to a file (replace 'output.json' with your desired output file name)
with open('data/output/valid_annotations.json', 'w') as output_file:
    json.dump(modified_data, output_file, indent=4)