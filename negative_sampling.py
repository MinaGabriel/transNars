import numpy as np
from data.utils import *
from data.TripletsDataset import *

class NegativeSampling:
    def __init__(self) -> None:
        pass

    @staticmethod
    def random_negative_sampling_r(positive_batch, dataset: TripletsDataset, negative_ratio=25): 
        
        num_positive_samples = positive_batch.shape[0]
        num_negative_samples = num_positive_samples * negative_ratio

        negative_batch = np.repeat(np.copy(positive_batch), negative_ratio, axis=0)

        random_entities = np.random.randint(dataset.num_entities, size=num_negative_samples)
        entity_indices = np.random.choice([0, 2], size=num_negative_samples)
        
        negative_batch[np.arange(num_negative_samples), entity_indices] = random_entities

        # Avoid sampling positive triples as negatives
        positive_set = set([tuple(triplet) for triplet in positive_batch.tolist()])
        negative_samples = set()

        for i in range(num_negative_samples):
            while tuple(negative_batch[i, :3]) in positive_set:
                random_entity = np.random.randint(dataset.num_entities)
                negative_batch[i, entity_indices[i]] = random_entity

            negative_samples.add(tuple(negative_batch[i, :3]))

        return negative_batch

def main():
    dataset_dir = "datasets/FB15K237"
    dataset = TripletsDataset(dataset_dir)
    for _, positive_batch in enumerate(DataLoader(dataset.train_dataset, batch_size=256, shuffle=True, num_workers=4)):
        negative_batch = NegativeSampling.random_negative_sampling_r(positive_batch, dataset)
        print(negative_batch.shape)
        print(negative_batch)

if __name__ == "__main__":
    main()