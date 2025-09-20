import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


def text_to_image(prompt: str):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )

    # MPS 지원 여부 확인
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pipe = pipe.to(device)

    url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

    init_image = load_image(url).convert("RGB")
    image = pipe(prompt, image=init_image).images

    return image

if __name__ == "__main__":
    prompt = "Give me a picture of a cat lying in bed"
    text_to_image(prompt)