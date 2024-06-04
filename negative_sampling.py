import numpy as np
from data.utils import *
from data import TripletsDataset

class NegativeSampling:
    def __init__(self) -> None:
        pass

    @staticmethod
    def random_negative_sampling_r(positive_batch, dataset: TripletsDataset, negative_ratio=25):
        """
        Random sampling : R
        Args: 
            positive_batch -> torch tensor
            dataset -> TripletsDataset 
            negative_ratio -> int : number of negative samples per positive sample
        Returns:
            Numpy array of negative samples:
            [s, r, o, index of repeated pos sample, 0 or 2 s or o was changed]
        """
        
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

        # Append indices and entity indices for tracking
        negative_batch = np.column_stack((negative_batch, [i % num_positive_samples for i in range(num_negative_samples)]))
        negative_batch = np.column_stack((negative_batch, entity_indices))

        return negative_batch

