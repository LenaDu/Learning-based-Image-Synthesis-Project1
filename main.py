from utils import *
import os
import skimage.io as skio
import skimage as sk
import argparse
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='please enter input directory with --indir and output directory with --outdir')
    parser.add_argument('--indir', type=str, help='input directory', required=True)
    parser.add_argument('--outdir', type=str, help='output directory', required=True)
    parser.add_argument('--textname', type=str, help='output text file name', required=False)

    args = vars(parser.parse_args())

    image_path = args['indir'] + '/'
    output_path = args['outdir'] + '/'
    text_file_name = args['textname']

    # print(image_path, output_path)
    if text_file_name:
        output_text_file = open(f'{text_file_name}.txt', 'a')
    else:
        output_text_file = open('output_data.txt', 'a')

    try:
        os.mkdir(output_path)
        print('output directory created')
    except:
        print('output directory already exists')

    images = os.listdir(image_path)

    for image in images:
        im = skio.imread(image_path + image)

        # convert to double (might want to do this later on to save memory)
        im = sk.img_as_float(im)

        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int)
        width = im.shape[1]

        # separate color channels
        b = im[:height]
        g = im[height: 2 * height]
        r = im[2 * height: 3 * height]

        # find alignment according to green channel

        # ** edge detection + ncc
        b_aligned, r_aligned, shift = pyramid(r, g, b, 'edge')
        im_out = np.dstack([r_aligned, g, b_aligned])

        # record the output data
        output_text = f''' [{image}]
        
To align blue channel and red channel with green channel, 
the alignment of blue channel at x is {shift[0]}, at y is {shift[1]};
the alignment of red channel at x is {shift[2]}, at y is {shift[3]};\n'''
        print(output_text)
        output_text_file.write(output_text)

        # get cropped image
        im_out = auto_cropping(im_out)

        # get white-balanced image
        im_out = auto_white_balance(im_out)

        # save the image
        imname = image.split('.')[0]
        fname = f'{output_path}{imname}.jpg'
        skio.imsave(fname, im_out)

    output_text_file.close()