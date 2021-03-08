from make_cubes import create_fake_cube


def main():
    noise_file = 'data/U6805.lmap.fits'
    new_cubes = []
    all_locs = []
    no_cubes = 10
    for i in range(no_cubes):
        print("Making cube %s "%i, "out of %s..."%no_cubes)
        gal_locs, new_cube = create_fake_cube(noise_file, set_wcs='B1950')
        si = gal_locs[0][0][0]
        new_cube.plot_slice(slice_i=si, sliced=False)
        new_cubes.append(new_cube.cube_data)
        all_locs.append(gal_locs)
    print("Done!")
    return new_cubes, all_locs

if __name__ == "__main__":
    main()