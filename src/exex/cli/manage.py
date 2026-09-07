"""Add, remove, or label experts in a MoE model checkpoint."""

import argparse
import os

from exex.manager import ExpertManager


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the input model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save the modified model")

    subparsers = parser.add_subparsers(dest="action", help="Action to perform", required=True)

    parser_remove = subparsers.add_parser("remove", help="Remove an expert from the model")
    parser_remove.add_argument("--expert_index", type=int, required=True,
                               help="Index of the expert to remove")

    parser_add = subparsers.add_parser(
        "add", help="Add a new expert to the model (cloned from an existing slot)")
    parser_add.add_argument("--clone_from", type=int, default=0,
                            help="Source expert index to clone")
    parser_add.add_argument("--label", type=str, default=None,
                            help="Label to give to the new expert")

    parser_label = subparsers.add_parser(
        "label", help="Add or update a label for an existing expert")
    parser_label.add_argument("--expert_index", type=int, required=True,
                              help="Index of the expert to label")
    parser_label.add_argument("--label_name", type=str, required=True, help="The label name")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    manager = ExpertManager(args.model_path)

    if args.action == "remove":
        manager.remove_expert(args.expert_index, args.output_dir)

    elif args.action == "add":
        new_idx = manager.clone_expert(args.clone_from, label=args.label)
        print(f"Cloned expert {args.clone_from} -> new slot {new_idx}")
        os.makedirs(args.output_dir, exist_ok=True)
        manager.model.save_pretrained(args.output_dir)
        manager.config.save_pretrained(args.output_dir)

    elif args.action == "label":
        manager.label_expert(args.expert_index, args.label_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving config with new labels to {args.output_dir}...")
        manager.config.save_pretrained(args.output_dir)
    return manager


if __name__ == "__main__":
    main()
