import numpy as np
from data.utils import *
from data.TripletsDataset import *
import random
from tqdm import tqdm

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
        positive_set = set(map(tuple, positive_batch.tolist()))
        negative_samples = set()

        for i in range(num_negative_samples):
            while tuple(negative_batch[i, :3]) in positive_set:
                random_entity = np.random.randint(dataset.num_entities)
                negative_batch[i, entity_indices[i]] = random_entity

            negative_samples.add(tuple(negative_batch[i, :3]))

        return negative_batch

    @staticmethod
    def corrupting_positive_instances_c(positive_batch, dataset: TripletsDataset, negative_ratio=25):
        num_positive_samples = positive_batch.shape[0]
        num_negative_samples = num_positive_samples * negative_ratio

        negative_samples = []
        for h, r, t in positive_batch:
            h, r, t = h.item(), r.item(), t.item()
            for _ in range(negative_ratio//2):
                #NOTE: r at the end to save to file as preprocessing step
                # Corrupt the head
                corrupted_h = random.choice(dataset.get_all_entity_ids())
                if (t in dataset.all_possible_hs and r in dataset.all_possible_hs[t]) and corrupted_h not in dataset.all_possible_hs[t][r]:
                    negative_samples.append((corrupted_h, t, r))

                # Corrupt the tail
                corrupted_t = random.choice(dataset.get_all_entity_ids())
                if h in dataset.all_possible_ts and r in dataset.all_possible_ts[h] and corrupted_t not in dataset.all_possible_ts[h][r]:
                    negative_samples.append((h, corrupted_t, r))

        negative_samples = np.array(negative_samples, dtype=np.int64)
        if len(negative_samples) > num_negative_samples and len(negative_samples) > 0:
            # Randomly remove extra samples
            indices = np.random.choice(len(negative_samples), num_negative_samples, replace=False)
            negative_samples = negative_samples[indices]
        elif len(negative_samples) < num_negative_samples and len(negative_samples) > 0:
            # Randomly repeat samples to reach the required number
            additional_samples = np.random.choice(len(negative_samples), num_negative_samples - len(negative_samples), replace=True)
            negative_samples = np.concatenate([negative_samples, negative_samples[additional_samples]])
        #print(negative_samples) 
        return negative_samples

    def save_negative_samples_to_file(negative_samples, file_path, total_elements):
        with open(file_path, 'w') as f:
            f.write(f"{total_elements}\n")
            for row in negative_samples:
                f.write(" ".join(map(str, row)) + "\n")   
                
                
if __name__ == "__main__":
    location = 'datasets/nations'
    file_name = 'neg_sam_n.txt'
    dataset = TripletsDataset(location)
    data_loader = DataLoader(
            dataset.train_dataset,
            batch_size=512, shuffle=False
        )
    all_negative_samples = []
    for _, data in enumerate(tqdm(data_loader, desc="Processing batches")):
        negative_samples = NegativeSampling.corrupting_positive_instances_c(data, dataset)
        negative_samples[[1, 2]] = negative_samples[[2, 1]]
        all_negative_samples.append(negative_samples)
        
    all_negative_samples = np.vstack(all_negative_samples) 
    NegativeSampling.save_negative_samples_to_file(all_negative_samples, location + '/' + file_name, len(all_negative_samples))
    