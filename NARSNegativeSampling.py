import numpy as np

from NARSKnowledgeGraph import NARSKnowledgeGraph
from data.utils import *
from data.TripletsDataset import *
import random
from tqdm import tqdm
import yaml


class NARSNegativeSampling:
    def __init__(self) -> None:
        pass

    @staticmethod
    def corrupting_positive_instances_NARS(
        NARS: NARSKnowledgeGraph,
        positive_batch,
        dataset: TripletsDataset,
        negative_ratio=25,
    ):
        num_positive_samples = positive_batch.shape[0]
        num_negative_samples = num_positive_samples * negative_ratio

        negative_samples = []
        for h, r, t in positive_batch:
            h, r, t = h.item(), r.item(), t.item()  # item from training set
            question_what_is_subject = NARS.TripletToQuestion(-1, t, r)
            judgment_results = NARS.DeriveAnswers(
                question_what_is_subject, negative_ratio // 2
            )
            if judgment_results is not None:
                for i in range(negative_ratio // 2):
                    if i >= len(judgment_results):
                        break
                    judgment = judgment_results[i]
                    # NOTE: r at the end to save to file as preprocessing step
                    # Corrupt the head
                    corrupted_h = NARS.entity_name_to_ID[
                        str(judgment.statement.get_subject_term())
                    ]
                    if (
                        t in dataset.all_possible_hs and r in dataset.all_possible_hs[t]
                    ) and corrupted_h not in dataset.all_possible_hs[t][r]:
                        negative_samples.append((corrupted_h, t, r))

            question_what_is_object = NARS.TripletToQuestion(h, -1, r)
            judgment_results = NARS.DeriveAnswers(
                question_what_is_object, negative_ratio // 2
            )
            if judgment_results is not None:
                for i in range(negative_ratio // 2):
                    if i >= len(judgment_results):
                        break
                    judgment = judgment_results[i]
                    # NOTE: r at the end to save to file as preprocessing step
                    # Corrupt the head
                    corrupted_t = NARS.entity_name_to_ID[
                        str(judgment.statement.get_predicate_term())
                    ]
                    if (
                        h in dataset.all_possible_ts
                        and r in dataset.all_possible_ts[h]
                        and corrupted_t not in dataset.all_possible_ts[h][r]
                    ):
                        negative_samples.append((h, corrupted_t, r))

                # Corrupt the tail

        negative_samples = np.array(negative_samples, dtype=np.int64)
        if len(negative_samples) > num_negative_samples and len(negative_samples) > 0:
            # Randomly remove extra samples
            indices = np.random.choice(
                len(negative_samples), num_negative_samples, replace=False
            )
            negative_samples = negative_samples[indices]
        elif len(negative_samples) < num_negative_samples and len(negative_samples) > 0:
            # Randomly repeat samples to reach the required number
            additional_samples = np.random.choice(
                len(negative_samples),
                num_negative_samples - len(negative_samples),
                replace=True,
            )
            negative_samples = np.concatenate(
                [negative_samples, negative_samples[additional_samples]]
            )
        # print(negative_samples)
        return negative_samples

    @staticmethod
    def save_negative_samples_to_file(negative_samples, file_path, total_elements):
        with open(file_path, "w") as f:
            f.write(f"{total_elements}\n")
            for row in negative_samples:
                f.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    location = 'datasets/benchmarks/FB15K237'
    output_file_name = 'NARS_generated_samples_' + str(location.split("/")[2]) + '.txt'
    print("Will output NARS samples to " + str(output_file_name))
    dataset = TripletsDataset(location)
    data_loader = DataLoader(dataset.train_dataset, batch_size=512, shuffle=False)

    NARS = NARSKnowledgeGraph(dataset_directory=location, silent_mode=True)
    NARS.LoadTrainingSet(-1)  # store Judgments for all the training set triples
    # NARS.RunTestSet()

    all_negative_samples = []
    for _, positive_batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        # get batch of negative samples
        negative_samples = NARSNegativeSampling.corrupting_positive_instances_NARS(
            NARS, positive_batch, dataset
        )

        if len(negative_samples) == 0:
            continue
        # append batch of negative samples to total list of negative samples
        negative_samples[[1, 2]] = negative_samples[[2, 1]]
        all_negative_samples.append(negative_samples)

    if len(all_negative_samples) == 0:
        print("No negative samples were generated. Exiting...")
    else:
        all_negative_samples = np.vstack(all_negative_samples)
        NARSNegativeSampling.save_negative_samples_to_file(
            all_negative_samples,
            location + "/" + output_file_name,
            len(all_negative_samples),
        )
