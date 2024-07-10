import torch
import torch.nn.functional as F


def normalized_cross_correlation(template, image):
    """
    Compute the normalized cross-correlation (NCC) between a template and an image.

    NCC is a measure of similarity between two signals/images. It is often used in template matching
    and image registration tasks.

    Parameters:
    template (torch.Tensor): The template tensor with shape (1, 1, Ht, Wt).
    image (torch.Tensor): The image tensor with shape (1, 1, Hi, Wi).

    Returns:
    torch.Tensor: A tensor representing the normalized cross-correlation map with shape (1, 1, Hi-Ht+1, Wi-Ht+1).
    """
    template_mean = torch.mean(template)
    padding = (template.shape[2] // 2, template.shape[3] // 2)
    image_mean = F.conv2d(image, torch.ones_like(template) / template.numel(), padding=padding)

    template = template - template_mean
    image = image - image_mean

    numerator = F.conv2d(image, template, padding=padding)
    denominator = torch.sqrt(F.conv2d(image ** 2, torch.ones_like(template), padding=padding) * torch.sum(template ** 2))

    return numerator / denominator


if __name__ == "__main__":
    test_template = torch.rand(1, 1, 5, 5)
    test_image = torch.rand(1, 1, 20, 20)
    test_ncc_map = normalized_cross_correlation(test_template, test_image)
    print(test_ncc_map)
