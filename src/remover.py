import cv2
import numpy as np

def remove_logo(image_path, logo_coordinates, algorithm='telea'):
    """
    Removes a logo from an image using inpainting.

    Args:
        image_path (str): The path to the input image.
        logo_coordinates (list): A list of four integers representing the
                                 bounding box of the logo, in the format
                                 [x1, y1, x2, y2].
        algorithm (str): The inpainting algorithm to use. Can be 'telea' or 'ns'.

    Returns:
        numpy.ndarray: The image with the logo removed.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask for the logo
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = logo_coordinates
    mask[y1:y2, x1:x2] = 255

    # Apply inpainting
    if algorithm == 'telea':
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    elif algorithm == 'ns':
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    else:
        raise ValueError("Invalid inpainting algorithm specified. Choose 'telea' or 'ns'.")

    return inpainted_image

import argparse

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Remove a logo from an image.')
    parser.add_argument('input_image', help='The path to the input image.')
    parser.add_argument('output_image', help='The path to the output image.')
    parser.add_argument('coords', type=int, nargs=4,
                        help='The coordinates of the logo to be removed (x1 y1 x2 y2).')
    parser.add_argument('--algorithm', type=str, default='telea',
                        choices=['telea', 'ns'],
                        help='The inpainting algorithm to use (telea or ns).')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Remove the logo from the image
    inpainted_image = remove_logo(args.input_image, args.coords, args.algorithm)

    # Save the resulting image
    cv2.imwrite(args.output_image, inpainted_image)

    print(f"Logo removed and image saved as {args.output_image}")
