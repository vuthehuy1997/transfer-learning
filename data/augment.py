import torchvision.transforms as transforms
import random

class RandomGaussBlur(object):
    """Random GaussBlurring on image by radius parameter.
    Args:
        radius (list, tuple): radius range for selecting from; you'd better set it < 2
    """
    def __init__(self, radius=None):
        if radius is not None:
            assert isinstance(radius, (tuple, list)) and len(radius) == 2, \
                "radius should be a list or tuple and it must be of length 2."
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius={0})'.format(self.radius)

def get_img_transform(config, augment=False):
    img_transform = transforms.Compose([
                transforms.Resize([config['max_height'], config['max_width']]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    img_augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0, hue=0),
                img_transform
            ])
    img_augment2 = transforms.Compose([
                RandomGaussBlur(radius=[-5.0, 5.0]),
                transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0, hue=0),
                img_transform
            ])
    if augment:
        return img_augment
    else:
        return img_transform