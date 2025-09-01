import numpy as np

# Simulated Iris dataset (first 30 samples from each class for simplicity)
# Format: [sepal_length, sepal_width, petal_length, petal_width]
# Labels: 0 = setosa, 1 = versicolor, 2 = virginica

def load_iris_data():
    # Normally you'd load from a file, but here's a simplified version
    from urllib.request import urlopen
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    raw = urlopen(url).read().decode("utf-8").strip().split("\n")
    data = []
    label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    for row in raw:
        if row:
            parts = row.split(",")
            features = list(map(float, parts[:4]))
            label = label_map[parts[4]]
            data.append(features + [label])
    return np.array(data)

# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Calculate Gini impurity
def gini(groups, classes):
    total = sum(len(group) for group in groups)
    gini_score = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = group[:, -1]
        for class_val in classes:
            p = np.sum(labels == class_val) / size
            score += p * p
        gini_score += (1 - score) * (size / total)
    return gini_score

# Split dataset
def split_dataset(dataset, feature, threshold):
    left = dataset[dataset[:, feature] < threshold]
    right = dataset[dataset[:, feature] >= threshold]
    return left, right

# Find best split
def get_best_split(dataset):
    classes = np.unique(dataset[:, -1])
    best_gini = float("inf")
    best_feature, best_threshold = None, None
    for feature in range(dataset.shape[1] - 1):
        for threshold in np.unique(dataset[:, feature]):
            left, right = split_dataset(dataset, feature, threshold)
            gini_score = gini([left, right], classes)
            if gini_score < best_gini:
                best_gini = gini_score
                best_feature, best_threshold = feature, threshold
    return best_feature, best_threshold

# Build tree recursively
def build_tree(dataset, depth=0, max_depth=3):
    labels = dataset[:, -1]
    if len(np.unique(labels)) == 1 or depth >= max_depth:
        value = int(np.bincount(labels.astype(int)).argmax())
        return Node(value=value)
    feature, threshold = get_best_split(dataset)
    left_data, right_data = split_dataset(dataset, feature, threshold)
    left = build_tree(left_data, depth + 1, max_depth)
    right = build_tree(right_data, depth + 1, max_depth)
    return Node(feature=feature, threshold=threshold, left=left, right=right)

# Predict using tree
def predict(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature] < node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)

# Main program
def main():
    data = load_iris_data()
    tree = build_tree(data)

    print("Enter measurements of the new Iris flower:")
    sepal_length = float(input("Sepal length (cm): "))
    sepal_width = float(input("Sepal width (cm): "))
    petal_length = float(input("Petal length (cm): "))
    petal_width = float(input("Petal width (cm): "))

    sample = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict(tree, sample)

    species = ["setosa", "versicolor", "virginica"]
    print(f"\n🌼 Predicted species: {species[prediction]}")

if __name__ == "__main__":
    main()
