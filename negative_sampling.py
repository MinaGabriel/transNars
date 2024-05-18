import numpy as np
from utils import *


class NegativeSampling():
    def __init__(self) -> None:
        pass

        """Random sampling : R
        Args: 
            positive batch data -> torch tensor
            dataset -> TripletDataset 
        Returns:
            Numpy array of negative samples:
            [s, r, o, index of repeated pos sample, 0 or 2 s or o was changed]
        Example: 
            >>> 
        """
        
    @staticmethod
    def random_negative_sampling_r(positive_batch, dataset: TripletsDataset):
        # Not Good: for the positive triple (Tom Cruise, starred in, Top Gun),
        # negative targets such as London or Mount Everest seem irrelevant.
        # TODO: add ratio I am assuming that the ration now is 1:1 (same batch size)
        negative_batch = np.repeat(np.copy(positive_batch), 1, axis=0)
        # how many samples in the batch
        num_samples = negative_batch.shape[0]
        random_entities = np.random.randint(
            dataset.config['num_entities'], size=num_samples)
        # if number of samples is 5 we get something like this [0,2,2,0,0]
        entity_indices = np.random.choice([0, 2], size=num_samples)
        # tmp_negative_batch[[0,1,2,3,4,5],[0,2,2,0,0]] = [5,9,13,4,5]
        negative_batch[np.arange(num_samples),
                       entity_indices] = random_entities
        # TODO: I am doing this but I have no idea what it is for for now
        negative_batch = np.column_stack(
            (negative_batch, [i % len(positive_batch) for i in range(len(negative_batch))]))
        negative_batch = np.column_stack(
            (negative_batch, entity_indices))  # index for S or O changed
        # FIXME: Do i need to check if K- don't exists in K+? the research paper is saying NO
        # because the probability of this happening is small and can be ignored.
        # I am going to call this FILTERING and do it her in the future.
        return negative_batch
