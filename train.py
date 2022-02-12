
from src.config import cfg
from src.utils import get_parser, set_seed
from src import trainer


def main():
    # Parse arguments
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Init seed for reproducibility
    set_seed(1234)

    trainer.train(cfg, do_evaluation=True)


if __name__ == '__main__':
    main()
