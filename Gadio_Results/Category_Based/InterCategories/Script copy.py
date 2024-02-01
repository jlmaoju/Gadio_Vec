import csv
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import json

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

def find_similarities(data, category_averages, top_n=100):
    all_similarities = defaultdict(lambda: defaultdict(list))

    for cat1, cat2 in combinations(category_averages.keys(), 2):
        # Compute similarity from cat1 to cat2
        similarities_cat1_to_cat2 = compute_similarity(data, cat1, category_averages[cat2], top_n)
        all_similarities[cat1][cat2] = similarities_cat1_to_cat2
        
        # Compute similarity from cat2 to cat1
        similarities_cat2_to_cat1 = compute_similarity(data, cat2, category_averages[cat1], top_n)
        all_similarities[cat2][cat1] = similarities_cat2_to_cat1

    return all_similarities


def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1 - dot_product / (norm_vec1 * norm_vec2)

def compute_similarity(data, category, target_avg_embedding, top_n):
    similarities = []
    for row in data:
        if row['Category'] == category:
            embedding = parse_embedding(row['Average Embedding'])
            distance = cosine_distance(embedding, target_avg_embedding)
            similarities.append((row['Title'], round(distance, 4)))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

def main(file_path):
    data = read_csv(file_path)
    category_averages = calculate_category_averages(data)
    all_similarities = find_similarities(data, category_averages)

    # Save the results to a JSON file
    with open('results.json', 'w', encoding='utf-8') as file:
        json.dump(all_similarities, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    file_path = 'G:\\Pet_Projects\\Gadio\\DB test\\average_embeddings_with_Category.csv'  # Update this path as needed
    main(file_path)
