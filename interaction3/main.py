import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="interaction3 entrypoint")
    parser.add_argument(
        "mode",
        choices=("collect", "train", "infer"),
        help="Pipeline mode to run",
    )
    args = parser.parse_args()

    if args.mode == "collect":
        print("Run: py -3 interaction3\\src\\data_collector.py")
    elif args.mode == "train":
        print("Run: py -3 interaction3\\src\\train_model.py")
    else:
        print("Run: py -3 interaction3\\src\\realtime_inference.py")


if __name__ == "__main__":
    main()

