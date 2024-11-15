import albumentations as A
from albumentations.pytorch import ToTensorV2

# Hyperparameters
LR = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_WORKERS = 0
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/ADE20K/images/training/"
TRAIN_MASK_DIR = "data/ADE20K/annotations/training/"
VAL_IMG_DIR = "data/ADE20K/images/validation/"
VAL_MASK_DIR = "data/ADE20K/annotations/validation/"
FN_FORMAT = ('jpg','png')
CLASSES = 150

MODEL_NAME = "segmentation/saved_models/ade20k_vB_.tar"

# Transforms
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
)
val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
)
blank_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2()
    ]
)