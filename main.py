from utils import (
    initialize_process,
    load_dataset,
    build_dataset,
    run_process,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
)


def main():
    args, run, weights_dir = initialize_process(
        NEPTUNE_API_KEY, NEPTUNE_PROJECT
    )

    datasets, labels, train_num, valid_num, test_num = load_dataset(
        name=args.name, data_dir=args.data_dir
    )
    train_set, valid_set, test_set = build_dataset(
        datasets, args.batch_size, args.img_size
    )

    run_process(
        args,
        labels,
        train_num,
        valid_num,
        test_num,
        run,
        train_set,
        valid_set,
        test_set,
        weights_dir,
    )


if __name__ == "__main__":
    main()
