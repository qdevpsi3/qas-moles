import pickle


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        loaded = pickle.load(f)
    print(f"Pickle loaded from {filepath}")
    return loaded


def save_pickle(instance, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(instance, f)
    print(f"Pickle saved to {filepath}")
