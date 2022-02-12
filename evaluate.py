
from src import evaluation
from src.config import cfg
from src.utils import get_parser, set_seed


def main():
    # Parse arguments
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Init seed for reproducibility
    set_seed(1234)

    # run evaluation
    evaluation.run_evaluation(cfg)


if __name__ == '__main__':
    main()
