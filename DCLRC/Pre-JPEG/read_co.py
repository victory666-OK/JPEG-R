import argparse

from torchvision.transforms.functional import to_pil_image
import os
import torchjpeg.codec

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by using it along with a custom DCT routine to decompress a JPEG")
parser.add_argument("input_file", help="Input image file, must be a JPEG")
#parser.add_argument("dct_img_file", help="dct_img_file file, must be a JPEG")
args = parser.parse_args()

for fileName in os.listdir(args.input_file):
    dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input_file + "//" + fileName)
    #im = to_tensor(Image.open(args.dct_img_file + "//" + fileName))

    #if im.shape[0] > 3:
    #    im = im[:3]

    
    print(quantization)