import csv
import json

csv_file = "data/Metadata_dev.csv"
json_file = "data/dev_data.json"

gender_mapping = {'F': 0, 'M': 1}

with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    data = []
    for row in csv_reader:
        row['id'] = str(row['id'])
        row["age"] = int(row["age"])
        row["label"] = int(row["label"])
        if 'gender' in row and row['gender'] in gender_mapping:
            row['gender'] = gender_mapping[row['gender']]
        data.append(row)

with open(json_file, mode='w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)