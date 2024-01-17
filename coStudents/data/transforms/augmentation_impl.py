import random
from PIL import ImageFilter
import torch
import torchvision.transforms as T
import numpy as np
import copy
from numpy import random as R


class GLT:
    def __init__(self, datasets='bdd'):
        self.gaussian = T.GaussianBlur(11, (0.1, 2.0))

        self.night_images_stats = {
            'bdd': [(0.165, 0.099), (0.152, 0.107), (0.141, 0.115)],
            'shift': [(0.105, 0.051), (0.080, 0.043), (0.058, 0.032)],
            'acdc': [(0.229, 0.063), (0.179, 0.050), (0.135, 0.062)],
        }
        self.night_prior = self.night_images_stats[datasets]

    def mask_img(self, img, cln_img):
        while R.random() > 0.5:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:, x1:x2, y1:y2] = cln_img[:, x1:x2, y1:y2]
        return img

    def adaptive_aug_param(self, img_, param_type):
        params = []
        img = copy.deepcopy(img_)
        img = img.float()
        for i in range(3):
            day_mean, day_std = torch.mean(img[i, :, :]), torch.std(img[i, :, :])
            night_mean, night_std = self.night_prior[i]
            offset = 0
            val = 0
            if param_type == 'brightness':
                offset = night_mean - day_mean
            else:
                offset = night_std / day_std
            val = torch.clamp(random.random() * offset, 0.2, 1.0)
            params.append(val)
        return params

    def aug(self, x, contrastive_feature=False):

        for sample in x:
            img = sample['image'].cuda()
            g_b_flag = True

            if R.random() > 0.5:
                img = self.gaussian(img)

            cln_img_zero = img.detach().clone()
            random_beta = random.uniform(0.2, 0.8)

            if R.random() > 0.4:
                cln_img = img.detach().clone()
                val = np.random.uniform(1.25, 5)
                img = T.functional.adjust_gamma(img.unsqueeze(0), val).squeeze(0)
                img = self.mask_img(img, cln_img)
                g_b_flag = False

            if R.random() > 0.55 or g_b_flag:
                cln_img = img.detach().clone()
                val = self.adaptive_aug_param(img, 'brightness')
                # Apply adaptive brightness separately to each channel
                for i in range(3):
                    img[i, :, :] = T.functional.adjust_brightness(img[i, :, :].unsqueeze(0),
                                                                  random_beta * val[i]).squeeze(0)
                img = self.mask_img(img, cln_img)

            if R.random() > 0.55:
                cln_img = img.detach().clone()
                val = self.adaptive_aug_param(img, 'contrast')
                # Apply adaptive contrast separately to each channel
                for i in range(3):
                    img[i, :, :] = T.functional.adjust_contrast(img[i, :, :].unsqueeze(0), val[i]).squeeze(0)
                img = self.mask_img(img, cln_img)
            img = self.mask_img(img, cln_img_zero)

            if R.random() > 0.5:
                n = torch.clamp(torch.normal(0, 0.1, img.shape), min=0).cuda()
                img = n + img.float()
                img = torch.clamp(img, max=1.0).type(torch.float)

            # # local transformation for contrastive feature
            try:
                if contrastive_feature:

                    lowlight_param = [R.uniform(1.5, 5) for _ in range(3)]
                    roi_mask = torch.ones(img.shape).cuda()
                    normalized_image = img

                    bboxes = sample['instances'].gt_boxes.tensor
                    num_selected_bboxes = round(len(bboxes) * 0.3) if len(bboxes) > 3 else 1
                    selected_bbox_indices = np.random.choice(len(bboxes), num_selected_bboxes, replace=False)

                    for bbox_idx in selected_bbox_indices:
                        bbox = bboxes[bbox_idx]
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if np.random.rand() > 0.5:
                            y_new = y1 + np.random.randint(y2 - y1)
                            h_new = (y2 - y_new) // 2
                            for i in range(3):
                                if h_new == 0:
                                    break
                                roi_mask[i, y_new:y_new + h_new, x1:x2] = random_beta * lowlight_param[i] * 0.5
                        else:
                            x_new = x1 + np.random.randint(x2 - x1)
                            w_new = (x2 - x_new) // 2
                            for i in range(3):
                                if w_new == 0:
                                    break
                                roi_mask[i, y1:y2, x_new:x_new + w_new] = random_beta * lowlight_param[i] * 0.5

                    lowlighted_roi = torch.pow(normalized_image, roi_mask)
                    lowlight_image_clapm = torch.clamp(lowlighted_roi, min=.0, max=1.0)
                    img = (lowlight_image_clapm * 255).type(torch.float)
            except Exception as e:
                print("augmentation error: ", e)

            sample['image'] = img.cpu()

        return x


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
