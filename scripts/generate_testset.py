"""Generate synthetic evaluation testset from a contract document.

Usage:
    # From a PDF contract:
    python scripts/generate_testset.py --file data/sample_contracts/nda.pdf

    # From a DOCX contract:
    python scripts/generate_testset.py --file data/sample_contracts/agreement.docx

    # Custom number of samples:
    python scripts/generate_testset.py --file contract.pdf --samples 20

    # Custom output path:
    python scripts/generate_testset.py --file contract.pdf --output data/my_testset.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation testset")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to contract document (PDF or DOCX)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of Q&A pairs to generate (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_eval/testset.json",
        help="Output path for testset JSON (default: data/synthetic_eval/testset.json)",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        # Try relative to project root
        file_path = PROJECT_ROOT / args.file
    if not file_path.exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    output_path = str(PROJECT_ROOT / args.output)

    # Load document
    from src.ingestion.loaders import load_document

    logger.info(f"Loading document: {file_path}")
    doc = load_document(str(file_path))
    logger.info(f"Loaded {doc.total_pages} pages from {doc.filename}")

    # Generate synthetic testset
    from src.evaluation.synthetic_data import generate_synthetic_testset

    samples = generate_synthetic_testset(
        contract_text=doc.full_text,
        n=args.samples,
        output_path=output_path,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("  Synthetic Testset Generated")
    print("=" * 50)
    print(f"  Source:    {doc.filename}")
    print(f"  Pages:    {doc.total_pages}")
    print(f"  Samples:  {len(samples)}")
    print(f"  Output:   {output_path}")
    print("=" * 50)

    print("\n  Sample questions:")
    for i, s in enumerate(samples[:5], 1):
        q = s.get("question", "")
        print(f"    {i}. {q}")
    if len(samples) > 5:
        print(f"    ... and {len(samples) - 5} more")

    print(f"\n  Now run evaluation:")
    print(f"    python scripts/run_eval.py --testset {args.output}\n")


if __name__ == "__main__":
    main()
