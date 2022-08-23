import random
import torch


def random_inpainting(image, cnt=5):
    """[summary]
    masking images with random window blocks, where values are randomly draw.
    code is adapted from Model Genesis: https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/utils.py
    Args:
        image ([ torch tensor]): a batch of images
        cnt: number of blocks to be drawn.
        probability ([type]): [description]

    Returns:
        [type]: [description]
    """
    n, c, h, w = image.size()
    newimage = image.clone()
    count = cnt
    assert 2 / 3 * h > 6 and 2 / 3 * w > 6, 'image is too small, got h:{},w:{}'.format(h, w)
    for i in range(n):
        cnt = count
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(h // 6, h // 3)
            block_noise_size_y = random.randint(w // 6, w // 3)
            noise_x = random.randint(3, h - block_noise_size_x - 3)
            noise_y = random.randint(3, w - block_noise_size_y - 3)
            newimage[i, :, noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y] = torch.rand(c, block_noise_size_x,
                                                                        block_noise_size_y, dtype=image.dtype, device=image.device) * 1.0
            cnt -= 1
    return newimage


def random_outpainting(image, cnt=5):
    """[summary]
    masking images with random noise while only pixels within random window blocks are preserved.
    code is adapted from Model Genesis: https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/utils.py
    Args:
        image ([ torch tensor]): a batch of images
        cnt: number of blocks to be drawn.
        probability ([type]): [description]

    Returns:
        [type]: [description]
    """
    n, c, h, w = image.size()
    orig_image = image.clone()
    assert 2 / 3 * h > 6 and 2 / 3 * w > 6, 'image is too small, got h:{},w:{}'.format(h, w)
    noise_image = torch.rand_like(image, device=image.device)
    count = cnt
    for i in range(n):
        cnt = count
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(h // 6, h // 3)
            block_noise_size_y = random.randint(w // 6, w // 3)
            noise_x = random.randint(3, h - block_noise_size_x - 3)
            noise_y = random.randint(3, w - block_noise_size_y - 3)
            noise_image[i, :, noise_x:noise_x + block_noise_size_x,
                        noise_y:noise_y + block_noise_size_y] = orig_image[i, :, noise_x:noise_x + block_noise_size_x,
                                                                           noise_y:noise_y + block_noise_size_y]
            cnt -= 1

    return noise_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = torch.zeros(1, 1, 12, 12)
    cutout_image = random_inpainting(image)
    print(cutout_image)
    plt.subplot(121)
    plt.imshow(cutout_image.data[0, 0])
    plt.subplot(122)
    maskout_image = random_outpainting(image)
    plt.imshow(maskout_image.data[0, 0])
    plt.show()
    plt.savefig('./random_window_masking.png')
