import numpy as np
from PIL import Image


def load_image(image_path : str) -> np.ndarray:
    '''
    Load an image from a file and convert it to a numpy array
    Parameters:
    image_path: the path to the image file
    return: the image as a numpy array
    '''

    img = Image.open(image_path).convert('RGB') # Open the image and convert it into the 3 RGB channels
    img_array = np.array(img) # Convert the image to a numpy array: height x width x channels
    
    return img_array


def downsample_image(image : np.ndarray, factor : int) -> np.ndarray:
    '''
    Downsample an image by a factor
    Parameters:
    image: the image to downsample
    factor: the size of the blocks to downsample by
    return: the downsampled image
    '''


    height, width, _ = image.shape # Get the height and width of the image

    if(height % factor != 0 or width % factor != 0):
        raise ValueError('The factor must divide the height and width of the image')

    new_height = height // factor  # Compute the reduced height
    new_width = width // factor    # Compute the reduced width

    # Create a new image with the reduced dimensions and 3 channels
    downsampled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Iterate over the new image and compute the average color for each block (factor*factor) from the original image
    for i in range(new_height):
        for j in range(new_width):
            # Get the block of pixels
            block = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            # Create a matrix (pixels * channels) and compute the average color of each channel for the block as an int8
            avg_color = block.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            downsampled_image[i, j] = avg_color

    return downsampled_image



def upscale_image(downsampled_image : np.ndarray, factor : int) -> np.ndarray:
    '''
    Upscale an image by a factor
    Parameters:
    downsampled_image: the image to upscale
    factor: the size of the blocks to upscale by
    return: the upscaled image
    '''

    # Repeat each row factor times along the vertical axis 
    upscaled_image = np.repeat(downsampled_image, factor, axis=0)
    # Repeat each column factor times along the horizontal axis 
    upscaled_image = np.repeat(upscaled_image, factor, axis=1)
    
    return upscaled_image