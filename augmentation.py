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

        img_size = TF._get_image_size(image) # Get Size of Input Image
        
        # Validation Augmentation
        if not train:

            # 1. Resizing
            size = resize_to # Adjust
            image = TF.resize(image, size)
            segmentation = TF.resize(segmentation, size)

            # # 2. Crop
            # out_size = resize_to # Adjust
            # image = TF.center_crop(image, out_size)
            # segmentation = TF.center_crop(segmentation, out_size)

        # Training Augmentation
        if train:
            # 1. Random resize crop
            # i = random.randint(0, img_size[1]-200) # -100 so that not (400, 400) chosen as left upper coordinate
            # j = random.randint(0, img_size[1]-200) # -100 so that not (400, 400) chosen as left upper coordinate
            # h = random.randint(200, img_size[1])
            # w = random.randint(200, img_size[1])
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(1., 1.))
            size = resize_to
            image = TF.resized_crop(image, top=i, left=j, height=h, width=w, size=size)
            segmentation = TF.resized_crop(segmentation, top=i, left=j, height=h, width=w, size=size)

            # 2. Rotation
            if random.random() > 0.5: # Adjust
                angle = random.randint(-180, 180) # Adjust
                image = TF.rotate(image, angle)
                segmentation = TF.rotate(segmentation, angle)
            
            # 3. Horizontal flip
            if random.random() > 0.5: # Adjust prob.
                image = TF.hflip(image)
                segmentation = TF.hflip(segmentation)

            # 4. Vertical flip
            if random.random() > 0.5: # Adjust prob.
                image = TF.vflip(image)
                segmentation = TF.vflip(segmentation)

            # 5. Random Grayscale
            # if random.random() > 0.5:
            #     image = TF.rgb_to_grayscale(image, num_output_channels=3)
            #     # not needed for segmentation mask        

            # 6. Gaussian Blur
            # if random.random() > 0.5:
            #     sigma = random.uniform(0.1, 0.5)
            #     kernel_size = random.randrange(3, 10, 2)
            #     image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
                # not needed for segmentation mask

            # 7. ColorJitter: Adjust brightness, contrast, saturation, hue
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor=brightness)

                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor=contrast)

                saturation = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor=saturation)

                # hue = random.uniform(0)
                # image = TF.adjust_hue(image, hue_factor=hue)
                #not needed for segmentation
            
            # 8. Sharpness
            if random.random() > 0.5:
                sharpness = random.uniform(0.8, 2)
                image = TF.adjust_sharpness(image, sharpness_factor=sharpness)
                # not needed for segmentation

            # # 9. Equalize
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

        return image, segmentation



def test_transform_fn(image: PIL or torch.tensor, resize_to=(400, 400)):
        """preprocessing for image in test:
        -> Input: PIL or tensor, Output: PIL or tensor"""

        img_size = TF._get_image_size(image) # Get Size of Input Image (Tuple)
        
        # Validation Augmentation
        # 1. Resizing
        size = resize_to # Adjust
        image = TF.resize(image, size)

        # # 2. Crop
        # out_size = resize_to # Adjust
        # image = TF.center_crop(image, out_size)

        # 3. ToTensor
        image = TF.to_tensor(image)

        # 4. Normalize
        image = TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return image