import argparse
from pathlib import Path
import argparse
from pathlib import Path
import glob
import os
import yaml
from PIL import Image
import shutil


orig_input_p = Path("input_orig")
orig_ref_p = Path("ref_orig")
scaled_out_p = Path("out_scaled")
input_p = Path("input")
ref_p = Path("ref")
out_p = Path("out")
MARINER_IMAGE_SIZE = 160


def check_if_all_images_are_same_size(dir_path):
    imgs = glob.glob(str(Path(dir_path, "*.png")))
    if len(imgs) == 0:
        return False
    w, h = Image.open(imgs[0]).size
    for img in imgs:
        w_, h_ = Image.open(img).size
        if w != w_ or h_ != h:
            return False
    return True


def resize(image, size):
    width, height = image.size
    if width < height:
        wpercent = size / float(width)
        w_size = size
        h_size = int((float(height) * float(wpercent)))
    else:
        hpercent = size / float(height)
        h_size = size
        w_size = int((float(width) * float(hpercent)))
    image = image.resize((w_size, h_size), Image.Resampling.LANCZOS)
    return image


def parse_configs(args):
    res = {}
    for config in args["configs"]:
        with open(config, "r") as stream:
            try:
                d = yaml.safe_load(stream)
                res = res | d
            except yaml.YAMLError as e:
                print(e)

    return res


def preprocess(data_root):
    # Backup input and ref images
    shutil.move(data_root / input_p, data_root / orig_input_p)
    shutil.move(data_root / ref_p, data_root / orig_ref_p)
    input_dir = data_root / input_p
    input_dir.mkdir(parents=True, exist_ok=True)

    input_imgs = glob.glob(str(Path(data_root, orig_input_p, "*.png")))

    # resize images to MARINER supported size
    for img in input_imgs:
        im = Image.open(img)
        im = resize(im, MARINER_IMAGE_SIZE)
        im.save(str(input_dir / Path(img).name))

    ref_imgs = glob.glob(str(Path(data_root, orig_ref_p, "*.png")))
    ref_dir = data_root / ref_p
    ref_dir.mkdir(parents=True, exist_ok=True)
    for img in ref_imgs:
        im = Image.open(img)
        im = resize(im, MARINER_IMAGE_SIZE)
        im.save(str(ref_dir / Path(img).name))


def postprocess(data_root):
    # scale up
    out_imgs = sorted(glob.glob(str(Path(data_root, out_p, "*.png"))))
    orig_in_imgs = sorted(glob.glob(str(Path(data_root, orig_input_p, "*.png"))))
    same_size = check_if_all_images_are_same_size(data_root / orig_input_p)
    script_path = "thirdparty/Real-ESRGAN/inference_realesrgan.py"
    if same_size:
        in_w, in_h = Image.open(orig_in_imgs[0]).size
        out_w, out_h = Image.open(out_imgs[0]).size
        scale_factor = in_w / out_w

        os.system(
            f"python {script_path} -i {data_root / out_p} -s {scale_factor} -o {data_root /scaled_out_p}"
        )
    else:
        for out_im, in_im in zip(out_imgs, orig_in_imgs):
            in_w, in_h = Image.open(in_im).size
            out_w, out_h = Image.open(out_im).size
            scale_factor = in_w / out_w

            os.system(
                f"python {script_path} -i {out_im} -s {scale_factor} -o {data_root /scaled_out_p}"
            )

    # Make sure the images are really the same size. Because of numerical errors they could be off by 1 pixel.
    scaled_out_imgs = sorted(glob.glob(str(Path(data_root, scaled_out_p, "*.png"))))
    for scaled_out_img, orig_input_im in zip(scaled_out_imgs, orig_in_imgs):
        out_w, out_h = Image.open(scaled_out_img).size
        orig_w, orig_h = Image.open(orig_input_im).size
        if out_w != orig_w or out_h != orig_h:
            out_clean = Image.open(scaled_out_img).resize(
                (orig_w, orig_h), Image.Resampling.LANCZOS
            )
            out_clean.save(scaled_out_img)

    # clean up
    shutil.rmtree(data_root / input_p)
    shutil.rmtree(data_root / ref_p)
    shutil.rmtree(data_root / out_p)

    shutil.move(data_root / orig_input_p, data_root / input_p)
    shutil.move(data_root / orig_ref_p, data_root / ref_p)
    shutil.move(data_root / scaled_out_p, data_root / out_p)


def run(args):
    print("Rescaling input images")
    preprocess(args["data_dir"])
    print("Predicting on low resolution images")
    os.system(
        f"python mariner/main.py predict -c {args['config']} --ckpt_path {args['ckpt_path']} --data_dir {args['data_dir']}"
    )
    print("Upscale predicted images with Real-ESRGAN")
    postprocess(args["data_dir"])


if __name__ == "__main__":
    if not Path("thirdparty/Real-ESRGAN/inference_realesrgan.py").exists():
        print(
            "This script requires Real-ESRGAN to be installed. Please download the submodule: git submodule update --init --recursive"
        )
    else:
        parser = argparse.ArgumentParser(
            description="Predict images with resolution much larger than 160. Downscales the images, predicts and upscales. Requiress Real-ESRGAN."
        )
        parser.add_argument("-c", "--config", type=Path)
        parser.add_argument("--ckpt_path", type=Path)
        parser.add_argument("--data_dir", type=Path)

        args, unknown = parser.parse_known_args()

        args = args.__dict__
        run(args)
