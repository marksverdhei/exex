"""Analyze MoE routing behavior across domains."""

import argparse
import os

from exex.analyzer import RoutingAnalyzer


def create_dummy_dataset():
    """
    Creates a small mock dataset for testing purposes if a real dataset isn't provided.
    Includes domains: eng, stem, coding, multilingual.
    """
    return [
        {"text": "The quick brown fox jumps over the lazy dog. This is standard English text.", "domain": "eng"},
        {"text": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.", "domain": "stem"},
        {"text": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)", "domain": "coding"},
        {"text": "Bonjour tout le monde. Hola mundo. Hallo Welt. Ciao a tutti.", "domain": "multilingual"},
        {"text": "To be or not to be, that is the question.", "domain": "eng"},
        {"text": "E = mc^2 is the equation of mass-energy equivalence proposed by Albert Einstein.", "domain": "stem"},
        {"text": "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()", "domain": "coding"},
        {"text": "El tiempo vuela como una flecha. La mosca de la fruta vuela como un plátano.", "domain": "multilingual"},
    ] * 10  # Multiply to have more samples


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the MoE model (local or HF hub)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="HF dataset name or local json(l) file with 'text' and 'domain' "
                             "columns. Uses dummy data if not provided.")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples per domain to analyze")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: cuda if available)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Truncation length per sample")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    analyzer = RoutingAnalyzer(args.model_path, device=args.device, max_length=args.max_length)

    if args.dataset_path:
        from datasets import load_dataset
        if os.path.isfile(args.dataset_path):
            dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        else:
            dataset = load_dataset(args.dataset_path, split="train")
        # Assuming dataset has 'text' and 'domain' columns
    else:
        print("No dataset provided, using internal multi-domain dummy dataset...")
        dataset = create_dummy_dataset()

    results = analyzer.analyze_dataset(
        dataset,
        text_col="text",
        domain_col="domain",
        max_samples_per_domain=args.max_samples
    )

    analyzer.print_associations(results)
    analyzer.find_correlations(results, threshold_ratio=0.75)
    return results


if __name__ == "__main__":
    main()
