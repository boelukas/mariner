import argparse
from pathlib import Path
import glob
import cv2
import numpy as np
import torch
import erqa
import sys
import json
import lpips
from tabulate import tabulate, tabulate_formats

sys.path.append(str(Path(sys.argv[0]).parent.parent))
from mariner.utils.scores import calculate_masa_psnr_ssim


class Metric:
    """Metric Baseclass"""

    def __init__(self, metric) -> None:
        self.metric = metric

    def score(self, image_1, image_2):
        "Score"
        score = self.metric(self.convert(image_1), self.convert(image_2))
        if torch.is_tensor(score):
            return score.item()
        else:
            return score

    def convert(self, image):
        "Converts the cv2 [H, W, BGR] [0:255] image to the input of the metric"
        return image


class PSNR_SSIM(Metric):
    def __init__(self) -> None:
        super().__init__(calculate_masa_psnr_ssim)

    def convert(self, image):
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image


class L1(Metric):

    def __init__(self) -> None:
        super().__init__(torch.nn.L1Loss())

    def convert(self, image):
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image


class PIPS(Metric):
    def __init__(self, net: str = "vgg") -> None:
        super().__init__(lpips.LPIPS(net=net))

    def convert(self, image):
        x = torch.from_numpy(image)
        x = x.type(torch.float32)
        x = x.permute(2, 0, 1)
        x = x[[2, 1, 0], :, :]  # BGR to RGB
        x = 2 * x / 255 - 1  # normalize between [-1, 1]

        return x


class ERQA(Metric):
    def __init__(self) -> None:
        super().__init__(erqa.ERQA())


def print_table(headers, data, out_file, table_format="simple"):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as out_file:
        out_file.write(tabulate(data, headers=headers, tablefmt=table_format))


def create_score_table(args):
    with open(args["table"]) as json_file:
        data_paths = json.load(json_file)

    headers = list(next(iter(data_paths.values())).keys())
    rows = []
    print("Start creating table")
    for row_name, datasets in data_paths.items():
        print("Row: " + row_name)
        row = [row_name]
        for dataset_name, paths in datasets.items():
            print("   Column: " + dataset_name)
            s = avg_scores(paths[0], paths[1])
            row.append(
                f"{s['PSNR']:.2f}/{s['SSIM']:.3f}/{s['MAE']:.3f}/{s['ERQAS']:.3f}/{s['LPIPS_VGG']:.3f}/{s['LPIPS_ALEX']:.3f}"
            )
        rows.append(row)
    if args["out"]:
        out_file = args["out"]
    else:
        out_file = Path(args["table"].parent, "scores.txt")
    print_table(
        headers=headers,
        data=rows,
        out_file=out_file,
        table_format=args["format"],
    )


def avg_scores(src_1_path, src_2_path):
    src_1_list = sorted(glob.glob(str(Path(src_1_path, "*"))))
    src_2_list = sorted(glob.glob(str(Path(src_2_path, "*"))))
    erqa_metric = erqa.ERQA()
    psnr_ssim_metric = PSNR_SSIM()
    erqa_metric = ERQA()
    lpips_alex_metric = PIPS(net="alex")
    psnrs = []
    ssims = []
    erqas = []
    lpips_alexs = []
    stats = []
    for image1, image2 in zip(src_1_list, src_2_list):
        im1_cv = cv2.imread(image1)  # B G R
        im2_cv = cv2.imread(image2)

        psnr, ssim = psnr_ssim_metric.score(im1_cv, im2_cv)
        erqa_im = erqa_metric.score(im1_cv, im2_cv)
        lpips_alex = lpips_alex_metric.score(im1_cv, im2_cv)

        stats.append([Path(image1).name, psnr, ssim, erqa_im, lpips_alex])
        psnrs.append(psnr)
        ssims.append(ssim)
        erqas.append(erqa_im)
        lpips_alexs.append(lpips_alex)

    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    std_psnr = np.std(psnrs)
    std_ssim = np.std(ssims)
    avg_erqas = np.mean(erqas)
    std_erqas = np.std(erqas)
    avg_lpips_alex = np.mean(lpips_alexs)
    std_lpips_alex = np.std(lpips_alexs)
    stats.append(["AVG", avg_psnr, avg_ssim, avg_erqas, avg_lpips_alex])
    stats.append(["STD", std_psnr, std_ssim, std_erqas, std_lpips_alex])

    return {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "ERQA": avg_erqas,
        "LPIPS_ALEX": avg_lpips_alex,
        "details": stats,
    }


def run_scores(args):
    res = avg_scores(args["images"][0], args["images"][1])

    headers = ["", "PSNR", "SSIM", "ERQA", "LPIPS_ALEX"]
    if args["out"]:
        out_file = args["out"]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as stats_file:
            stats_file.write(tabulate(res["details"], headers=headers))
    print(tabulate(res["details"][-2:], headers=headers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates the evaluation metrics PSNR, SSIM, ERQA, LPIPS_ALEX. Works either on two directories --images or on a json formatted table layout."
    )
    parser.add_argument("--images", "-i", nargs=2, type=Path, required=False)
    parser.add_argument("--table", "-t", type=Path)
    parser.add_argument(
        "--format", "-f", type=str, choices=tabulate_formats, default="latex"
    )
    parser.add_argument("--out", "-o", type=Path, help="outdir/filename.txt")

    args, unknown = parser.parse_known_args()

    args = args.__dict__
    if args["table"]:
        create_score_table(args)
    else:
        run_scores(args)
