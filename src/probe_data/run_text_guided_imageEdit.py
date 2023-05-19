# Perform different editing as per object_subset.json using editing models:
# editing functions: pix2pix_edit, null_text_edit and imagic_edit

# Command to run this file:
# CUDA_VISIBLE_DEVICES="0,1" python run_text_guided_imageEdit.py --edit_model imagic --model_id CompVis/stable-diffusion-v1-4 --object_json_file object_subset.json

import torch 
from PIL import Image
import open_clip
import torchvision 
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets  import VisionDataset 
from typing import Any, Callable, List, Optional, Tuple
import os, shutil
import random 
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import numpy as np 
import torch.nn.functional as F
import argparse
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, AdamW 
from tqdm import tqdm
from torch import nn
import json
import subprocess
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Accelerate
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
import itertools 
import PIL 
import math 

CACHE_DIR = "/cmlscratch/shweta12/diffusion_checkpoints/"

# Transformation Function
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


# COCO-Detection 
class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root 
        # Load the labels
        self.labels = []
        self.paths = []
        self.total_labels = []

        # Category ids
        cat_ids_total = self.coco.getCatIds()
        
        cat_ids = self.coco.loadCats(self.coco.getCatIds())
        category_mappings = {cat_id:[] for cat_id in cat_ids_total}

        for sup_cat in cat_ids:
            category_mappings[sup_cat['id']] = sup_cat['name']
        
 

        # Across image_ids ===> extract the labels for that particular image 
        for img_id in self.ids:
            self.paths.append(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name']))
            ann_ids = self.coco.getAnnIds(imgIds = img_id)

            # Comes with segmentation masks, bounding box coordinates, image_classes (Segmentation classes)
            target = self.coco.loadAnns(ann_ids)
            #print(target)
            #print(img_id)
            curr_label = [category_mappings[segment['category_id']] for segment in target]
            self.labels.append(curr_label)
            self.total_labels += curr_label 
        

        # Loading the correct labels
        print(f'Loading the labels corresponding to each dataset .. ')

        # Unique Labels
        self.unique_labels = list(set(self.total_labels))
        

    # Load_image
    def _load_image(self, id: int) -> Image.Image:
        # print(id)
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    # Get_item_
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)        

        return 
    
    # Search
    def search(self, class_label=None):
        # Class label
        print(f'Selected Class Labels: {class_label}')

        # Select the images which have the corresponding class_image 
        self.relevant_ids = []

        # iterate
        c = 0
        num_images = 0
        # Iterate through the image_ids
        for img_id in self.ids:
            curr_label = self.labels[c]
                
            # Current_label
            if len(curr_label) < 2 and class_label in curr_label:
                # Save the image in the directory
                num_images += 1

                # Current image
                curr_image = self._load_image(img_id)

                # Save 
                save_path = './images/img_' + str(img_id) + '.png'

                # Save the current image
                curr_image.save(save_path)

            # Update 
            c += 1
        
        # Update the number of images 
        print(f'Number of images: {num_images}')


        return  
    
    # Function to save the images 
    def save(self, curr_ids, curr_path):
        # Iterate through the current ids
        for id_ in curr_ids:
            curr_image = self._load_image(id_)
            curr_image.save(curr_path + str(id_) + '.png')
        
        return 
    
    # Length
    def __len__(self) -> int:
        return len(self.ids)


# Function for pix2pix edit
def pix2pix_edit(img_dict, edit_data, model_id):
    # Fill in the pix2pix
    print(f'Into pix2pix')

    # Model ID
    if model_id is None:
        model_id = "timbrooks/instruct-pix2pix"    

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir = CACHE_DIR, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    #images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    img_ids = list(img_dict.keys())
    
    # Local instructions
    local_instructions = []

    # Iterate through the image_ids
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]

        # Edit instructions
        edit_instructions = []

        # Classes
        classes = list(edit_data.keys())
        class_curr = None 
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())

            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple 
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                
                class_curr = cls_ 
        

        # Current class
        instructions = edit_instructions[0]
        
        # attributes
        attributes = list(instructions.keys())
        
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:
                    local_prompt = "Add a " + attribute_local 
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
                                        
            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "Change the shape of the " + class_curr + " to " + attribute_local
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
            
            # object_replacement
            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt =  "replace "+ class_curr+ " by " + attribute_local
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
            # alter_parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = attribute_local + " to " + class_curr 
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
            # texture
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt =  "change texture of "+ class_curr+ " to "+ attribute_local+ " " +class_curr 
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
            # color
            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = "change color of "+class_curr+" to "+attribute_local
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))
            # pose
            elif attr == 'pose':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = "change pose of "+class_curr+" "+attribute_local
                    local_instructions.append((img_id, class_curr, attr, local_prompt, attribute_local))

            # To add -- other parts
            
            # Create the edited images
            
        # 

    # Local attributes instructions
    # local_instructions : contains tuple of the form (img_id, attribute, local_prompt (to be given to pix2pix), sub-attribute)
    # Total Number of Edits
    # Save the edited image as (img_id_attr_attribute_local.png)

    # Save-path
    save_path = './edited_images/pix2pix_results/'    
    os.makedirs(save_path, exist_ok=True)

    # Higher value will ensure faithfulness to the original image
    image_guidance_scales = [1, 1.5, 2.0, 2.5]

    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    guidance_scale = [6.5, 7.0,  8.0, 8.5] # 7.5,9.5

    # DONE: Add the guidance scale parameters; Save for a range of guidance scale parameters
    for img_guide in image_guidance_scales:
        for text_guide in guidance_scale:
            # Iterate through the local instructions
            for instr in local_instructions:
                curr_image_id = instr[0]
                class_curr = instr[1]
                curr_attribute = instr[2]
                curr_prompt = instr[3]
                curr_attribute_local = instr[4]

                os.makedirs(save_path+class_curr, exist_ok=True)
                os.makedirs(save_path+class_curr+'/'+curr_attribute, exist_ok=True)
                os.makedirs(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local, exist_ok=True)

                # Current image
                curr_image = img_dict[curr_image_id]
                # Save the unedited version
                curr_image.save(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local + '/'+ str(curr_image_id) + '_unedited.png')

                # Edited image
                image_edit = pipe(curr_prompt, image=curr_image, num_inference_steps=50, image_guidance_scale=img_guide, guidance_scale=text_guide).images[0]

                # Save the image
                image_edit.save(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local +'/'+ str(curr_image_id)+'_'+str(img_guide) + '_' + str(text_guide) + '.png')

    return 


# Function for Null-Text inversion edit 
def null_text_edit(img_dict, edit_data, model_id):
    if model_id is None: model_id = 'CompVis/stable-diffusion-v1-4'
    print(f'Into Null Text Inversion..')
        
    # Prepare Instructions:
    # Step 2: Prepare edit-instructions for each image in img_dict:
    img_ids = list(img_dict.keys())
    # Local instructions
    local_instructions = []
    # Iterate through the image_ids
    
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]
        # Edit instructions
        edit_instructions = []
        # Classes
        classes = list(edit_data.keys())
        class_curr = None
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())
            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                class_curr = cls_ # current object category
        
        instructions = edit_instructions[0]
        # attributes
        attributes = list(instructions.keys())
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:                    
                    local_prompt = "A photo of a " + class_curr +"[SEP] A photo of a " + class_curr + " and a " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a " + class_curr+"[SEP] A photo of a " + class_curr + " in the shape of " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    local_prompt = class_curr+"[SEP]"+attribute_local + " to the " + class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Texture
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ class_curr+"[SEP] A photo of a "+ class_curr +" with "+ attribute_local+ " texture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Color
            elif attr == 'color':
                for attribute_local in instruction_attr:
                    local_prompt = "A photo of a "+ class_curr+"[SEP]"+"A photo of a "+ attribute_local + " "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))            

            # Replace objects
            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))
            # Pose
            elif attr == 'pose':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {class_curr} {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            
    ## local_instructions : 
    ## contains tuple of the form (img_id, attribute, class_curr, local_prompt (to be given to imagic), sub-attribute)
    ## Here local_prompt consists of "source prompt" [SEP] "target prompt"


    ## Step 2: Load Baseline LDM_STABLE
    from diffusers import DDIMScheduler
    from null_text_code.ptp_utils import AttentionStore, text2image_ldm_stable, make_controller
    from null_text_code.null_inversion import NullInversion

    save_path = './edited_images/null_text_results'    
    os.makedirs(save_path, exist_ok=True)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=CACHE_DIR, scheduler=scheduler).to("cuda")
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    for instr in tqdm(local_instructions):
        curr_image_id = instr[0]
        curr_attribute = instr[1]
        class_curr = instr[1]
        curr_prompt = instr[3]
        curr_attribute_local = instr[4]
        
        # create a subfolder with "attribute name"
        os.makedirs(save_path+'/'+class_curr, exist_ok=True)
        os.makedirs(save_path+'/'+class_curr+'/'+curr_attribute, exist_ok=True)
        os.makedirs(save_path+'/'+class_curr+'/'+curr_attribute+'/'+curr_attribute_local, exist_ok=True)
        os.makedirs(save_path+'/'+class_curr+'/'+curr_attribute+'/'+curr_attribute_local+'/'+str(curr_image_id), exist_ok=True)

        # Curr image
        curr_image = img_dict[curr_image_id]
        # Save the unedited version        
        curr_image.save(save_path+'/'+class_curr+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) + '_unedited.png')

        # Edited image
        source_prompt = curr_prompt.split('[SEP]')[0]
        target_prompt = curr_prompt.split('[SEP]')[1]
        NUM_DDIM_STEPS = 50
        GUIDANCE_SCALE = 7.5
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        ## Method: Prompt To Prompt Based Image Generation via Inversion
        null_inversion = NullInversion(ldm_stable, NUM_DDIM_STEPS, GUIDANCE_SCALE, device)    
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(curr_image, source_prompt, offsets=(0,0,0,0), verbose=True)
        assert uncond_embeddings is not None, "WARNING: uncond_embeddings are NONE even after null-text optimization."

        prompts = [source_prompt]
        controller = AttentionStore()

        image_inv, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings)
        prompts = [target_prompt]

        ## HYPER-PARAMETERS FOR NULL TEXT inference.
        cross_replace_steps = {'default_': .8}
        self_replace_steps = .6

        if curr_attribute=='object_replacement':
            is_replace_controller=True
        else:
            is_replace_controller=False
        
        blend_word = (((class_curr,), (curr_attribute_local, class_curr))) # for local edit (regional edits)
        eq_params = {"words": (curr_attribute_local), "values": (2)}  # amplify attention to the words curr_attribute_local such as: "silver" by *2 
        
        controller = make_controller(prompts, tokenizer, is_replace_controller, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        edited_images, _ = text2image_ldm_stable(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)    
        # Save the image
        image_edit = edited_images[0]
        edit_img_save_path = save_path+'/'+class_curr+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) +'_alpha' + str(alpha) + '_text' + str(text_guide) + '.png'
        image_edit.save(edit_img_save_path)
        torch.cuda.empty_cache()    
    return 


# Function for Imagic Edit
def imagic_edit(img_dict, edit_data, model_id):
    print(f'Into Imagic')
    save_path = './edited_images/imagic_results/'    
    os.makedirs(save_path, exist_ok=True)

    # Step 1: load pipeline 
    # Local imports:
    from diffusers import DDIMScheduler
    pipe = DiffusionPipeline.from_pretrained(
                            model_id,
                            cache_dir=CACHE_DIR,
                            safety_checker=None,
                            use_auth_token=True,
                            custom_pipeline="imagic_stable_diffusion",
                            scheduler = DDIMScheduler(\
                                        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",\
                                        clip_sample=False, set_alpha_to_one=False)
                            )
    pipe.to("cuda")

    # Step 2: Prepare edit-instructions for each image in img_dict:
    img_ids = list(img_dict.keys())
    # Local instructions
    local_instructions = []
    # Iterate through the image_ids
    
    for img_id in img_ids:
        # Load image which needs to be passed to the diffusion model
        img_curr = img_dict[img_id]
        # Edit instructions
        edit_instructions = []
        # Classes
        classes = list(edit_data.keys())
        class_curr = None
        for cls_ in classes:
            curr_tuple = list(edit_data[cls_].keys())
            # Image_id
            if img_id in curr_tuple:
                # Search the instructions from the current tuple
                instructions = edit_data[cls_][img_id]
                edit_instructions.append(instructions)
                class_curr = cls_ # current object category
        
        # make directory for an object category:        
        os.makedirs(save_path+class_curr, exist_ok=True)

        instructions = edit_instructions[0]
        # attributes
        attributes = list(instructions.keys())
        # Iterate through the attributes
        for attr in attributes:
            # Get instricution list
            instruction_attr = instructions[attr]['to']
            # Object addition
            if attr == 'object_addition':
                for attribute_local in instruction_attr:                    
                    # local_prompt = "A " + class_curr + " and a " + attribute_local
                    local_prompt = "A photo of a " + class_curr + " and a " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Shape
            elif attr == 'shape':
                for attribute_local in instruction_attr:                    
                    # local_prompt = "A " + class_curr + " in the shape of a " + attribute_local
                    local_prompt = "A photo of a " + class_curr + " in the shape of " + attribute_local
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Alter parts
            elif attr == 'alter_parts':
                for attribute_local in instruction_attr:
                    # local_prompt = attribute_local + " to the " + class_curr
                    local_prompt = attribute_local + " to the " + class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Texture
            elif attr == 'texture':
                for attribute_local in instruction_attr:
                    # local_prompt = "a "+ class_curr +" with "+ attribute_local+ " texture"
                    local_prompt = "A photo of a "+ class_curr +" with "+ attribute_local+ " texture"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Color
            elif attr == 'color':
                for attribute_local in instruction_attr:
                    # local_prompt = "a "+ attribute_local + " "+ class_curr
                    local_prompt = "A photo of a "+ attribute_local + " "+ class_curr
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))            

            # Replace objects
            elif attr == 'object_replacement':
                for attribute_local in instruction_attr:
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            # Pose
            elif attr == 'pose':
                change_action_dict = {'sit': 'sitting', 'run': 'running', 'hit': 'hitting', 'jump': 'jumping',
                            'stand': 'standing', 'lay': 'laying'}
                for attribute_local in instruction_attr:
                    attribute_local = change_action_dict[attribute_local]
                    local_prompt = f"A photo of a {class_curr} [SEP] A photo of a {class_curr} {attribute_local}"
                    local_instructions.append((img_id, attr, class_curr, local_prompt, attribute_local, class_curr))

            else:
                raise ValueError(f"Attribute {attr} is not implemented!")
            
    ## local_instructions : 
    ## contains tuple of the form (img_id, attribute, local_prompt (to be given to imagic), sub-attribute)

    # Step 3: Load the images one by one and finetune the model for each image.
    # Step 3.1: Run inference with difference hyper-params. 
    # alpha used for interpolation of our learned embedding with the new embedding.
    alphas = [0.9, 1, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8]
    # Higher guidance scale will ensure faithfulness to the text, with a drawback of reducing the fidelity of the image
    guidance_scale = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    
    # Fix seed for generation
    generator = torch.Generator("cuda").manual_seed(0)

    # DONE: Add the guidance scale parameters; Save for a range of guidance scale parameters
    # Iterate through the local instructions
    for instr in tqdm(local_instructions):
        curr_image_id = instr[0]
        curr_attribute = instr[1]
        class_curr = instr[2]
        curr_prompt = instr[3]
        curr_attribute_local = instr[4]
        
        # create a subfolder with "attribute name"
        os.makedirs(save_path+class_curr+'/'+curr_attribute, exist_ok=True)
        os.makedirs(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local, exist_ok=True)
        os.makedirs(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local+'/'+str(curr_image_id), exist_ok=True)

        # Curr image
        curr_image = img_dict[curr_image_id]
        # Save the unedited version        
        curr_image.save(save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) + '_unedited.png')

        # Edited image
        _ = pipe.train(curr_prompt, image=curr_image, generator=generator)

        ## Once the pipeline is trained, run inference with different alpha and text guidance scales.
        for alpha in alphas:
            for text_guide in guidance_scale:                                                                             
                image_edit = pipe(num_inference_steps=50, alpha=alpha, guidance_scale=text_guide).images[0]
                # Save the image
                edit_img_save_path = save_path+class_curr+'/'+curr_attribute+'/'+curr_attribute_local + '/'+str(curr_image_id)+'/'+ str(curr_image_id) +'_alpha' + str(alpha) + '_text' + str(text_guide) + '.png'
                image_edit.save(edit_img_save_path)
        
        torch.cuda.empty_cache()
        
    return 


# Function to create edit 
# img_dict: {"img_id": img}
# edit_data: original dictionary storing the information from the json file
def create_edit(img_dict, edit_data, args):
    # Models: instructpix2pix, SINE, Null-text inversion, Dreambooth, Textual inversion, Imagic, Instabooth
    # Except instructpix2pix -- all other models require fine-tuning on the original image
    if args.edit_model == 'pix2pix':
        pix2pix_edit(img_dict, edit_data, args.model_id)
    
    elif args.edit_model == 'null_text':
        null_text_edit(img_dict, edit_data, args.model_id)
        
    elif args.edit_model == 'imagic':
        imagic_edit(img_dict, edit_data, args.model_id)

    else:
        print(f'Error .... wrong choice of editing model ')
    
    return 


# Function to edit images
# cap : dataset_class
# edit_data : json containing the data
def perform_edit(cap, edit_data, args):
    # Relevant classes
    relevant_classes = list(edit_data.keys())

    # Image dictionary which stores the image-id and extracted image
    img_dict = {}
    # iterate through coco class
    for coco_class in relevant_classes:
        # Image indexes
        img_indexes_data = edit_data[coco_class]
        
        # Class images
        class_images = []
        # image indexes
        img_indexes = list(img_indexes_data.keys())
        
        # Iterate through the images
        for img_id in img_indexes:
            img_ = cap._load_image(int(img_id))
            class_images.append(img_)
            img_dict[img_id] = img_ 
        
        # Create a dictionary of {"img_id": img}        

    # Function to create edits
    create_edit(img_dict, edit_data, args)

    return 


# Main function to create the editing loader
def main():

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_label", default='sink', type=str, required=False, help="Class Label")

    parser.add_argument("--edit_model", default='pix2pix', type=str, required=False, help="Diffusion Model to use for editing.\
        options = ['pix2pix', 'null_text', 'imagic']")
    
    parser.add_argument("--model_id", default=None, type=str, required=False, help="Model id/checkpoint to load for a given edit_model")
    parser.add_argument("--object_json_file", default='object_subset.json', type=str, required=False, help="json file to read attribute-edit formats.")

    args = parser.parse_args()

    # Open the json file
    f = open(args.object_json_file)

    # Load the data
    edit_data = json.load(f)
    
    # Define the COCO dataset
    cap = CocoDetection(root = '/fs/cml-datasets/coco/images/train2017',
                            annFile = '/fs/cml-datasets/coco/annotations/instances_train2017.json',
                            transform =_transform(224))#transforms.PILToTensor())

    # Function which performs the edit given the json references
    perform_edit(cap, edit_data, args) 

    return 


# If main function:
if __name__ == "__main__":
    # Main function
    main()


