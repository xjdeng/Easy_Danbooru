from PIL import Image
import numpy as np
import torch
import tqdm
import torch.hub
import os

import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()

model_url = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt"

cached_model = torch.hub.get_dir() + '/checkpoints/' + model_url.split('/')[-1]
if os.path.exists(cached_model):
    print(f"Model found in cache: {cached_model}")
    mod = torch.load(cached_model)
else:
    print(f"Model not found in cache, downloading...")
    mod = torch.hub.load_state_dict_from_url(model_url)
model.load_state_dict(mod)

model.eval()

if torch.cuda.is_available():
    model.cuda()
    model.half()

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

x = torch.from_numpy(a)

if torch.cuda.is_available():
    x = x.to("cuda", dtype=torch.half)

# first run
with torch.no_grad():
    y = model(x)[0].detach().cpu().numpy()

# measure performance
for n in tqdm.tqdm(range(10)):
    with torch.no_grad():
        model(x)

for i, p in enumerate(y):
    if p >= 0.5:
        print(model.tags[i], p)