from argparse import ArgumentParser
import os
import numpy as np
import cv2
import imghdr
import itertools
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


def list_subtree(root_dir, recursive):
    if os.path.isfile(root_dir):
        return [root_dir]
    elif recursive:
        subtree_files = []
        for dir, _, files in os.walk(root_dir):
            for file in files:
                subtree_files.append(os.path.abspath(os.path.join(dir, file)))
        return subtree_files
    else:
        all_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir)]
        return [path for path in all_paths if os.path.isfile(path)]


def is_image(file_path):
    return imghdr.what(file_path) is not None


def load_image(img_path):
    with open(img_path, "rb") as stream:
        img_bytes = bytearray(stream.read())
        img_np = np.asarray(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def center_crop(img):
    crop_size = np.min(img.shape[0: 2])
    h_range = (img.shape[0] // 2 - crop_size // 2, img.shape[0] // 2 + crop_size // 2)
    w_range = (img.shape[1] // 2 - crop_size // 2, img.shape[1] // 2 + crop_size // 2)
    return img[h_range[0]:h_range[1], w_range[0]:w_range[1], :]


def create_tiles(img_dir, tile_size, out_dir):
    tile_counter = itertools.count()
    print("Listing images sub-tree...")
    ls = list_subtree(img_dir, recursive=True)
    img_files = [f for f in tqdm(ls, desc="Filtering non-image files") if is_image(f)]
    for img_f in tqdm(img_files, desc="Creating tiles"):
        try:
            img = load_image(img_f)
            cropped_img = center_crop(img)
            tile = cv2.resize(cropped_img, (tile_size, tile_size))
            tile_path = os.path.join(out_dir, f"tile_{next(tile_counter):08}" + os.path.splitext(img_f)[1])
            cv2.imwrite(tile_path, tile)
        except:
            print(f"Failed to create tile from {img_f}")


def load_tiles(tiles_dir, tile_size):
    ls = list_subtree(tiles_dir, recursive=True)
    img_files = [f for f in tqdm(ls, desc="Filtering non-image files from tiles directory") if is_image(f)]

    tiles_list = []
    for img_f in img_files:
        img = load_image(img_f)
        if np.array_equal(img.shape, [tile_size, tile_size, 3]):
            tiles_list.append(img)

    return np.array(tiles_list, dtype=np.uint8)


def image_to_tiles(img, tiles_dim):
    if (img.shape[0] % tiles_dim[0] != 0) or (img.shape[1] % tiles_dim[1] != 0):
        raise ValueError("Image tiles dimension must be divisible by the image dimension")

    n_height = int(img.shape[0] / tiles_dim[0])
    n_width = int(img.shape[1] / tiles_dim[1])

    tiles_array = np.zeros((n_height, n_width, tiles_dim[0], tiles_dim[1], img.shape[2]), dtype=img.dtype)
    for h_ind in range(n_height):
        for w_ind in range(n_width):
            h_start, h_end = h_ind * tiles_dim[0], (h_ind + 1) * tiles_dim[0]
            w_start, w_end = w_ind * tiles_dim[1], (w_ind + 1) * tiles_dim[1]
            tiles_array[h_ind, w_ind] = img[h_start:h_end, w_start:w_end, :]

    return tiles_array


def compare_tiles(img_tiles, tiles_set):
    s = ssim(imageA, imageB)


def center_to_external_idx_2d(sizes):
    raise NotImplementedError()


def tiles_to_image(tiles):
    raise NotImplementedError()


def create_mosaic(src_img, tiles):
    src_tiles = image_to_tiles(src_img, tiles.shape[1:3])
    tiles_similarity_mat = compare_tiles(src_tiles, tiles)
    idx_order = center_to_external_idx_2d(src_tiles.shape[0:2])
    out_tiles = np.zeros(src_tiles.shape, dtype=np.uint8)
    for (h_ind, w_ind) in idx_order:
        matching_tile_ind = np.argmax(tiles_similarity_mat[h_ind, w_ind, :])
        out_tiles[h_ind, w_ind] = tiles[matching_tile_ind]
        tiles_similarity_mat[:, :, matching_tile_ind] = 0
    out_img = tiles_to_image(out_tiles)
    return out_img


def save_image(img, out_path):
    raise NotImplementedError()


def _main():
    tiles_dir = args.tiles_dir if args.tiles_dir is not None else os.path.join(args.out_dir, "tiles")
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
        create_tiles(args.img_dir, args.tile_size, tiles_dir)

    img_array = load_image(args.src_image)
    mosaic_dim = (int(np.round(img_array.shape[0] * args.scaling_factor / args.tile_size) * args.tile_size),
                  int(np.round(img_array.shape[1] * args.scaling_factor / args.tile_size) * args.tile_size))
    img = cv2.resize(img_array, mosaic_dim)

    tiles = load_tiles(tiles_dir)

    mosaic_img = create_mosaic(img, tiles)
    save_image(mosaic_img, os.path.join(args.out_dir, "mosaic.jpg"))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("src_image", help="Path to source image")
    arg_parser.add_argument("out_dir", help="Path to output directory")
    arg_parser.add_argument("--img-dir", help="Path to image directory (image can contain different sizes)")
    arg_parser.add_argument("--tiles-dir", help="Path to tiles directory (all have the same size)")
    arg_parser.add_argument("--tile-size", type=int, default=50, help="Tiles size")
    arg_parser.add_argument("--scaling-factor", type=float, default=10,
                            help="The output image will be this many time larger than the original image")
    args = arg_parser.parse_args()

    if args.tiles_dir is None and args.img_dir is None:
        raise ValueError("Either images or tiles directory much be provided")

    _main()
