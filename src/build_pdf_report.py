"""Build a single PDF report from images in a run analysis folder.

Usage:
  python -m src.build_pdf_report --run_dir <run_dir>
"""
from __future__ import annotations

import argparse
import os
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_pdf", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    analysis = os.path.join(args.run_dir, 'analysis')
    out_pdf = args.out_pdf or os.path.join(analysis, 'report.pdf')

    # collect PNGs in analysis folder
    imgs = []
    for root, _, files in os.walk(analysis):
        for f in sorted(files):
            if f.lower().endswith('.png'):
                imgs.append(os.path.join(root, f))

    if not imgs:
        raise SystemExit('No PNGs found to build report')

    pil_imgs = [Image.open(p).convert('RGB') for p in imgs]
    first, rest = pil_imgs[0], pil_imgs[1:]
    first.save(out_pdf, save_all=True, append_images=rest)
    print('Wrote PDF report to', out_pdf)


if __name__ == '__main__':
    main()
