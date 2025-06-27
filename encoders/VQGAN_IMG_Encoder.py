from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer

# Load VQGAN configuration and checkpoint (pretrained on ImageNet)
vqgan_config = OmegaConf.load("vqgan_imagenet.yaml")  # user needs the model config
vqgan = VQModel(**vqgan_config.model.params)
vqgan.load_state_dict(torch.load("vqgan_imagenet.ckpt")["state_dict"], strict=False)
vqgan = vqgan.eval().to('cuda')

def encode_image(img_tensor):
    """
    img_tensor: [3 x H x W], normalized to [-1,1]
    Returns flattened codebook indices or feature vector.
    """
    with torch.no_grad():
        z, _, [_, _, indices] = vqgan.encode(img_tensor.unsqueeze(0).to('cuda'))
        # z: latent feature, indices: codebook indices [1, H', W']
        features = z.flatten(1)  # [1, latent_dim]
    return features
