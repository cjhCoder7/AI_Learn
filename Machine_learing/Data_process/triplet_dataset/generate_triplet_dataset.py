import random
import tqdm


def find_samples(pos_data, neg_data, idx):
    pos_samples = []
    neg_samples = []
    for pos_idx, pos_per_data in enumerate(pos_data):
        if idx == pos_idx:
            continue
        else:
            pos_samples.append(pos_per_data)
    for neg_idx, neg_per_data in enumerate(neg_data):
        neg_samples.append(neg_per_data)

    pos_sample = random.choice(pos_samples)
    neg_sample = random.choice(neg_samples)

    return pos_sample, neg_sample


def get_contrastive_data(easy_data, hard_data):
    contrastive_data = []

    # for idx, per_data in enumerate(tqdm.tqdm(hard_data)):
    #     pos_sample, neg_sample = find_samples(hard_data, easy_data, idx)
    #     contrastive_data.append(
    #         {
    #             "anchor": per_data["question"],
    #             "positive": pos_sample["question"],
    #             "negative": neg_sample["question"],
    #         }
    #     )

    for idx, per_data in enumerate(tqdm.tqdm(hard_data)):
        pos_sample, neg_sample = find_samples(hard_data, easy_data, idx)
        contrastive_data.append(
            {
                "anchor": per_data["text"],
                "positive": pos_sample["text"],
                "negative": neg_sample["text"],
            }
        )
    return contrastive_data
