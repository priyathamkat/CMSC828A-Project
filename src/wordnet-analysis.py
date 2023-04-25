import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from absl import app, flags
from accelerate import Accelerator
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel, ViTModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 32, "Batch size for computing distances.")
flags.DEFINE_integer(
    "num_batches", 1000, "Number of batches of images to compute distances for."
)
flags.DEFINE_integer("num_workers", 1, "Number of workers for the dataloader.")
flags.DEFINE_enum(
    "model", "resnet50", ["resnet50", "clip", "dino"], "Model to use for computing distances."
)
flags.DEFINE_string("cache_dir", "../cache", "Cache directory.")

def cache_factory(function):
    @functools.wraps(function)
    def cached_function(cache_dir, *args, **kwargs):
        cache_dir = Path(cache_dir)
        cached_result_path = cache_dir / f"{function.__name__}.npy"
        try:
            return np.load(cached_result_path)
        except FileNotFoundError:
            result = function(*args, **kwargs)
            cached_result_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cached_result_path, result)
            return result
    return cached_function

def similarity_matrix_factory(function):
    @functools.wraps(function)
    def similarity_matrix(num_classes, synsets):
        result = np.zeros((num_classes, num_classes))
        total = num_classes * (num_classes - 1) // 2
        with tqdm(total=total, desc=f"Computing {function.__name__}", leave=False) as pbar:
            for i in range(num_classes):
                for j in range(i):
                    result[i, j] = function(synsets[i], synsets[j])
                    pbar.update(1)
        return result + result.T - np.diag(np.diag(result))
    return similarity_matrix

def main(_):
    
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="wordnet-analysis",
        config=FLAGS.flag_values_dict(),
        init_kwargs={"wandb": {"name": FLAGS.model}},
    )
   
    if FLAGS.model == "resnet50":
        model_name = "microsoft/resnet-50"

        base_model = AutoModel.from_pretrained(model_name)
        base_model = accelerator.prepare(base_model)
        def model(*args, **kwargs):
            return base_model(*args, **kwargs).pooler_output.squeeze()

        base_preprocessor = AutoImageProcessor.from_pretrained(model_name)
        def preprocess(*args, **kwargs):
            inputs = base_preprocessor(*args, **kwargs, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"][0]
            return inputs
    elif FLAGS.model == "clip":
        model_name = "openai/clip-vit-base-patch32"

        base_model = CLIPVisionModel.from_pretrained(model_name)
        base_model = accelerator.prepare(base_model)
        def model(*args, **kwargs):
            return base_model(*args, **kwargs).pooler_output

        base_preprocessor = AutoImageProcessor.from_pretrained(model_name)
        def preprocess(images):
            inputs = base_preprocessor(images=images, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"][0]
            return inputs
    elif FLAGS.model == "dino":
        model_name = "facebook/dino-vitb16"

        base_model = ViTModel.from_pretrained(model_name)
        base_model = accelerator.prepare(base_model)
        def model(*args, **kwargs):
            return base_model(*args, **kwargs).last_hidden_state.mean(dim=1)

        base_preprocessor = AutoImageProcessor.from_pretrained(model_name)
        def preprocess(images):
            inputs = base_preprocessor(images=images, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"][0]
            return inputs
    else:
        raise ValueError(f"Model {FLAGS.model} not supported.")

    imagenet = ImageNet(root="/fs/cml-datasets/ImageNet/ILSVRC2012", transform=preprocess)
    imagenet_idx_to_wnid = {v: k for k, v in imagenet.wnid_to_idx.items()}

    def imagenet_class_to_wordnet_synset(class_idx):
        wnid = imagenet_idx_to_wnid[class_idx]
        pos, offset = wnid[0], int(wnid[1:])
        return wordnet.synset_from_pos_and_offset(pos, offset)
    
    num_classes = 1000
    synsets = [imagenet_class_to_wordnet_synset(i) for i in range(num_classes)]

    def get_similarity_matrix(similarity):
        return cache_factory(similarity_matrix_factory(similarity))(FLAGS.cache_dir, num_classes, synsets)

    lch_similarity_matrix = get_similarity_matrix(wordnet.lch_similarity)
    wup_similarity_matrix = get_similarity_matrix(wordnet.wup_similarity)
    path_similarity_matrix = get_similarity_matrix(wordnet.path_similarity)

    dataloader = DataLoader(
        imagenet,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )
    dataloader = accelerator.prepare(dataloader)

    num_samples_per_batch = FLAGS.batch_size * (FLAGS.batch_size - 1) // 2
    num_samples = FLAGS.num_batches * num_samples_per_batch
    
    lch_similarities = np.zeros(num_samples)
    wup_similarities = np.zeros(num_samples)
    path_similarities = np.zeros(num_samples)
    image_cosine_similarities = np.zeros(num_samples)

    idx = 0

    iv, jv = np.tril_indices(FLAGS.batch_size, k=-1)

    for batch, _ in zip(dataloader, trange(FLAGS.num_batches)):

        image_inputs, labels = batch
        labels = labels.cpu().numpy()

        with torch.no_grad():
            features = model(**image_inputs)
            features /= features.norm(dim=1, keepdim=True) + 1e-8
            cosine_similarity = features @ features.T
            cosine_similarity = cosine_similarity.cpu().numpy()

        liv, ljv = labels[iv], labels[jv]

        lch_similarities[idx:idx + num_samples_per_batch] = lch_similarity_matrix[liv, ljv]
        wup_similarities[idx:idx + num_samples_per_batch] = wup_similarity_matrix[liv, ljv]
        path_similarities[idx:idx + num_samples_per_batch] = path_similarity_matrix[liv, ljv]
        image_cosine_similarities[idx:idx + num_samples_per_batch] = cosine_similarity[iv, jv]
        
        idx += num_samples_per_batch
    
    fig = px.scatter(x=image_cosine_similarities, y=lch_similarities)
    accelerator.log({"lch_cosine": fig})
    plt.clf()
    fig = px.scatter(x=image_cosine_similarities, y=wup_similarities)
    accelerator.log({"wup_cosine": fig})
    plt.clf()
    fig = px.scatter(x=image_cosine_similarities, y=path_similarities)
    accelerator.log({"path_cosine": fig})
    plt.clf()

    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)
