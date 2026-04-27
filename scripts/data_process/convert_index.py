import os
from data.mmap_dataset import repack_properties
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_txt', type=str, required=True, help='Path to the index.txt file')
    parser.add_argument('--output_npy', type=str, default='', help='Path to save the output index.npy file')
    args = parser.parse_args()

    repack_properties(
        txt_file=args.index_txt,
        convert_file=args.output_npy if args.output_npy != '' else None
    )
