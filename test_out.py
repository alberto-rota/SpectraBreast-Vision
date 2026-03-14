import torch
import sys
sys.path.append("/home/arota/spectra/mast3r")
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R

device = "cuda"
model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)

images_norm = torch.rand(2, 3, 480, 640, device=device)
views = []
for i in range(2):
    views.append({
        "img": images_norm[i:i+1],
        "true_shape": torch.tensor([[480, 640]], dtype=torch.int32, device=device),
        "idx": i,
        "instance": str(i)
    })

pairs = [(views[i], views[i]) for i in range(2)]
with torch.no_grad():
    output = inference(pairs, model, device, batch_size=2, verbose=False)
    
print(type(output))
if isinstance(output, dict):
    print(output.keys())
elif isinstance(output, list):
    print("List of length:", len(output))
    print("Type of element:", type(output[0]))
