import os

# the real ViT runs on CPU here, dinov2.layers reads this at import
os.environ.setdefault("XFORMERS_DISABLED", "1")
