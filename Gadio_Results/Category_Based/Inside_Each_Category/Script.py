import csv
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def read_csv(file_path):
    with open(file_path, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def parse_embedding(embedding_str):
    return np.array([float(x) for x in embedding_str.split(',')])

def calculate_category_averages(data):
    category_embeddings = defaultdict(list)
    for row in data:
        embedding = parse_embedding(row['Average Embedding'])
        category_embeddings[row['Category']].append(embedding)
    category_averages = {cat: np.mean(embeds, axis=0) for cat, embeds in category_embeddings.items()}
    return category_averages

def find_nearest_and_farthest_titles(data, category_averages, top_n=10):
    category_distances = defaultdict(list)
    
    for row in tqdm(data, desc="Processing"):
        embedding = parse_embedding(row['Average Embedding'])
        category = row['Category']
        distance = np.linalg.norm(embedding - category_averages[category])

        category_distances[category].append((row['Title'], distance))

    category_results = {}
    for category, distances in category_distances.items():
        sorted_distances = sorted(distances, key=lambda x: x[1])
        category_results[category] = {
            'nearest': sorted_distances[:top_n],
            'farthest': sorted_distances[-top_n:]
        }
    
    return category_results

def main(file_path):
    data = read_csv(file_path)
    category_averages = calculate_category_averages(data)
    results = find_nearest_and_farthest_titles(data, category_averages)

    for category, result in results.items():
        print(f"Category: {category}")
        print("  Nearest Titles:")
        for title, distance in result['nearest']:
            print(f"    {title} (Distance: {distance})")
        print("  Farthest Titles:")
        for title, distance in result['farthest']:
            print(f"    {title} (Distance: {distance})")
        print()

if __name__ == "__main__":
    file_path = 'G:\\Pet_Projects\\Gadio\\DB test\\average_embeddings_with_Category.csv'  # Update this path as needed
    main(file_path)
