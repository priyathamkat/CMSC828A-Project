import functools
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
from transformers import AutoImageProcessor, AutoModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 32, "Batch size for computing distances.")
flags.DEFINE_integer(
    "num_batches", 1000, "Number of batches of images to compute distances for."
)
flags.DEFINE_integer("num_workers", 1, "Number of workers for the dataloader.")
flags.DEFINE_enum(
    "model", "resnet50", ["resnet50"], "Model to use for computing distances."
)

def cache_factory(function):
    @functools.wraps(function)
    def cached_function(cache_dir, *args, **kwargs):
        cache_dir = Path(cache_dir)
        cached_result_path = cache_dir / f"{function.__name__}.npy"
        try:
            return np.load(cache_dir)
        except FileNotFoundError:
            result = function(*args, **kwargs)
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(result, cached_result_path)
            return result
    return cached_function

def similarity_matrix_factory(function):
    @functools.wraps(function)
    def similarity_matrix(num_classes, synsets):
        result = np.zeros((num_classes, num_classes))
        total = num_classes * (num_classes - 1) // 2
        with tqdm(total=total, desc=f"Computing {function.__name__} similaritites", leave=False) as pbar:
            for i in range(num_classes):
                for j in range(i):
                    result[i, j] = function(synsets[i], synsets[j])
                    pbar.update(1)
        return result + result.T - np.diag(np.diag(result))
    return similarity_matrix

@cache_factory
@similarity_matrix_factory
def wn_lch(synset_1, synset_2):
    return wordnet.lch_similarity(synset_1, synset_2)

@cache_factory
@similarity_matrix_factory
def wn_wup(synset_1, synset_2):
    return wordnet.wup_similarity(synset_1, synset_2)

@cache_factory
@similarity_matrix_factory
def wn_path(synset_1, synset_2):
    return wordnet.path_similarity(synset_1, synset_2)

def main(_):
    if FLAGS.model == "resnet50":
        model_name = "microsoft/resnet-50"

    model = AutoModel.from_pretrained(model_name)
    preprocess = AutoImageProcessor.from_pretrained(model_name)

    imagenet = ImageNet(root="/fs/cml-datasets/ImageNet/ILSVRC2012")
    imagenet_idx_to_wnid = {v: k for k, v in imagenet.wnid_to_idx.items()}

    def imagenet_class_to_wordnet_synset(class_idx):
        wnid = imagenet_idx_to_wnid[class_idx]
        pos, offset = wnid[0], int(wnid[1:])
        return wordnet.synset_from_pos_and_offset(pos, offset)

    dataloader = DataLoader(
        imagenet,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        collate_fn=lambda x: zip(*x),
    )
    iterator = iter(dataloader)

    num_samples = FLAGS.num_batches * FLAGS.batch_size * (FLAGS.batch_size - 1) // 2
    wordnet_similarities = np.zeros(num_samples)
    image_cosine_similarities = np.zeros(num_samples)

    idx = 0

    for _ in trange(FLAGS.num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        images, labels = batch
        synsets = [imagenet_class_to_wordnet_synset(label) for label in labels]

        with torch.no_grad():
            image_inputs = preprocess(images, return_tensors="pt")
            features = model(**image_inputs).pooler_output.squeeze()
            features /= features.norm(dim=1, keepdim=True) + 1e-8
            cosine_similarity = features @ features.T
            cosine_similarity = cosine_similarity.cpu().numpy()

        for i in range(FLAGS.batch_size):
            for j in range(i):
                wordnet_similarities[idx] = wordnet.lch_similarity(
                    synsets[i], synsets[j]
                )
                image_cosine_similarities[idx] = cosine_similarity[i, j]
                idx += 1


if __name__ == "__main__":
    app.run(main)
