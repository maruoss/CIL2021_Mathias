# libraries
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import PIL

def transform_fn(image: PIL or torch.tensor or np.array, segmentation: PIL or torch.tensor, train=True, resize_to=(400, 400)):
        """preprocessing for image, segmentation in training:
        -> Input: PIL or tensor or np.array, Output: PIL or tensor"""

        # 0. Convert numpy to PIL (transforms only work on PIl or tensors)
        if isinstance(image, np.ndarray):
            image = TF.to_pil_image(image)  
            segmentation = TF.to_pil_image(segmentation)

        input_img_size = TF._get_image_size(image) # Get Size of Input Image as list
        
        # Validation Augmentation
        if not train:

            # 1. Resizing # Either resize here or predict via patching (in trainer.py file)
            image = TF.resize(image, resize_to)
            segmentation = TF.resize(segmentation, resize_to)

            # # 2. Crop
            # out_size = resize_to # Adjust
            # image = TF.center_crop(image, out_size)
            # segmentation = TF.center_crop(segmentation, out_size)

        # Training Augmentation
        if train:
            
            # 0. Mirror padding for rotation. About ~100 padding on each side needed for 45 degree rotation case.
            image = transforms.functional.pad(image, padding=100, padding_mode="reflect")
            segmentation = transforms.functional.pad(segmentation, padding=100, padding_mode="reflect")

            # 1. Rotation + Center Crop
            # if random.random() > 0.5: # Adjust
            # angle = random.randint(-180, 180) # Adjust
            angle = random.choice([-180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
            # Center crop
            image = TF.center_crop(image, input_img_size)
            segmentation = TF.center_crop(segmentation, input_img_size)

            # # 2. Random resize crop: simulate "zoom-in" of satellite images # COMMENTED OUT: TEST IMAGES ARE RATHER ZOOMED "OUT"!
            # i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(1.0, 1.0), ratio=(1., 1.))
            # image = TF.resized_crop(image, top=i, left=j, height=h, width=w, size=input_img_size)  # Resize to input image size
            # segmentation = TF.resized_crop(segmentation, top=i, left=j, height=h, width=w, size=input_img_size)

            # 3. Random crop (to predict on patches)
            i, j, h, w = transforms.RandomCrop.get_params(image, resize_to) # crop to desired size
            image = transforms.functional.crop(image, i, j, h, w)
            segmentation = transforms.functional.crop(segmentation, i, j, h, w)
            
            # 4. Horizontal flip
            if random.random() > 0.5: # Adjust prob.
                image = TF.hflip(image)
                segmentation = TF.hflip(segmentation)

            # 5. Vertical flip
            if random.random() > 0.5: # Adjust prob.
                image = TF.vflip(image)
                segmentation = TF.vflip(segmentation)

            # 6. Random Grayscale
            if random.random() > 0.5:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
                # not needed for segmentation mask        

            # 7. Gaussian Blur
            if random.random() > 0.5:
                sigma = random.uniform(0.1, 0.5)
                kernel_size = random.randrange(3, 10, 2)
                image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
                # not needed for segmentation mask

            # 8. ColorJitter: Adjust brightness, contrast, saturation, hue
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor=brightness)

                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor=contrast)

                saturation = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor=saturation)

                # hue = random.uniform(0)
                # image = TF.adjust_hue(image, hue_factor=hue)
                # # not needed for segmentation
            
            # 9. Sharpness
            if random.random() > 0.5:
                sharpness = random.uniform(0.8, 2)
                image = TF.adjust_sharpness(image, sharpness_factor=sharpness)
                # not needed for segmentation

            # # 10. Equalize
            # if random.random() > 0.5:
            #     image = TF.equalize(image)
            #     # not needed for segementation

        # To both train and validation:
        # 3. ToTensor
        image = TF.to_tensor(image)
        segmentation = TF.to_tensor(segmentation)

        # 4. Normalize
        image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # not needed for segmentation

        # 5. Random Erasing
        # Randomly erase a square of scale 0.1, value=None -> random values.
        i, j, h, w, v = transforms.RandomErasing.get_params(image, scale=(0.1, 0.1), ratio=(1, 1), value=None)
        image = TF.erase(image, i=i, j=j, h=h, w=w, v=v)


        return image, segmentation



def test_transform_fn(image: PIL or torch.tensor, resize_to=None):
        """preprocessing for image in test:
        -> Input: PIL or tensor, Output: PIL or tensor"""

        img_size = TF._get_image_size(image) # Get Size of Input Image (Tuple)
        
        if resize_to is not None:
            # 0. Resizing
            size = resize_to # Adjust
            image = TF.resize(image, size)

        # 1. ToTensor
        image = TF.to_tensor(image)

        # 2. Normalize
        image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return image