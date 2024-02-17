import json
from pathlib import Path
from typing import Optional
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers.activations import QuickGELUActivation
import math
from einops.layers.torch import Rearrange
import einops
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
from path import Path as pathpy

try:
    from VisionModel import VisionModel
except ImportError:
	from .VisionModel import VisionModel

class JoytagModel:
	def __init__(self):
		checkpoints_dir = f"{torch.hub.get_dir()}/checkpoints"
		pathpy(checkpoints_dir).mkdir_p()
		model_dir = f"{checkpoints_dir}/joytag"
		pathpy(model_dir).mkdir_p()
		modelfile = f"{model_dir}/model.safetensors"
		configfile = f"{model_dir}/config.json"
		tagfile = f"{model_dir}/top_tags.txt"
		if not pathpy(modelfile).exists():
			torch.hub.download_url_to_file("https://huggingface.co/fancyfeast/joytag/resolve/main/model.safetensors", modelfile)
		if not pathpy(configfile).exists():
			torch.hub.download_url_to_file("https://huggingface.co/fancyfeast/joytag/raw/main/config.json", configfile)
		if not pathpy(tagfile).exists():
			torch.hub.download_url_to_file("https://huggingface.co/fancyfeast/joytag/raw/main/top_tags.txt", tagfile)
		model = VisionModel.load_model(model_dir)
		model.eval()
		if torch.cuda.is_available():
			model = model.to('cuda')
		with open(tagfile, 'r') as f:
			self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
		self.model = model
	
	@staticmethod		
	def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
        # Pad image to square
		image_shape = image.size
		max_dim = max(image_shape)
		pad_left = (max_dim - image_shape[0]) // 2
		pad_top = (max_dim - image_shape[1]) // 2

		padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
		padded_image.paste(image, (pad_left, pad_top))

        # Resize image
		if max_dim != target_size:
			padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
        
        # Convert to tensor
		image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

        # Normalize
		image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

		return image_tensor

	@torch.no_grad()
	def predict(self, image, THRESHOLD = 0.4):
		if isinstance(image, str):
			image = Image.open(image)
		image_tensor = JoytagModel.prepare_image(image, self.model.image_size)
		batch = {
			'image': image_tensor.unsqueeze(0).to('cuda'),
		}

		with torch.amp.autocast_mode.autocast('cuda', enabled=True):
			preds = self.model(batch)
			tag_preds = preds['tags'].sigmoid().cpu()
	
		scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
		predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
		tag_string = ', '.join(predicted_tags)

		return tag_string, scores	
