from make_cubes import create_training_set
import argparse


def main(mos_dir, gal_dir, out_dir):
    # Create mock data for training set
    create_training_set(mos_dir)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Finding")
    parser.add_argument('mos_dir', type=str, default="data/mosaics",
     help='The directory of the noise cubes to insert the mock galaxies into')
    parser.add_argument('gal_dir', type=str, default='data/mock_gals',
     help='The directory of the mock galaxy cubes')
    parser.add_argument('out_dir', type=str, default="data/training",
     help='The output directory of the synthetic cubes')
    args = parser.parse_args()

    main(args.mos_dir, args.gal_dir, args.out_dir)