import pathlib

# TODO: set a path to cache files?
cache_path = pathlib.Path(__file__).parent / "cache"

def main(index: int, overwrite: bool = False):
    cache_file = cache_path / f"output_{index}.hdf5"

    if cache_file.exists() and not overwrite:
        print(f"Cache file already exists for index={index}")
        return 0
    
    print(f"Running index={index}")
    # DO STUFF


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, required=True)
    parser.add_argument("--overwrite", "-o", action="store_true", default=False)
    args = parser.parse_args()
    main(args.index, overwrite=args.overwrite)









