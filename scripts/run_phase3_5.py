#!/usr/bin/env python
"""Run Phase 3.5: Embedding Drift Detection Analysis"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.analysis.drift_detection import DriftDetector
from src.analysis.drift_visualization import DriftVisualizer
from src.dataset.loader import DatasetLoader
from src.embeddings.cache import EmbeddingCache
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.utils.config import ensure_directories, load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run drift detection analysis."""
    setup_logging(log_level="INFO")
    ensure_directories()

    logger.info("=" * 70)
    logger.info("PHASE 3.5: EMBEDDING DRIFT DETECTION ANALYSIS")
    logger.info("=" * 70)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_config = load_config("config/dataset_config.yaml")
    dataset_path = Path(dataset_config["dataset"]["processed_path"])

    loader = DatasetLoader()
    dataset = loader.load_json(dataset_path)
    texts, labels = loader.extract_features_labels(dataset)
    class_names = loader.get_class_names(dataset)

    logger.info(f"Loaded {len(texts)} examples")

    # Initialize embedder (using Sentence Transformer for speed)
    logger.info("Initializing embedder...")
    embedder = SentenceTransformerEmbedder()

    # Initialize drift detector
    logger.info("Initializing drift detector...")
    detector = DriftDetector(embedder)

    # Analyze dataset for drift
    logger.info("Analyzing embedding drift for all queries...")
    logger.info("This will take a few minutes (need to embed each query twice)...")

    results = detector.analyze_dataset(texts, labels, class_names)

    # Calculate category statistics
    logger.info("Calculating category statistics...")
    stats = detector.calculate_category_stats(
        results["drift_scores"], results["labels"], class_names
    )

    # Print stats
    stats_report = detector.format_stats_report(stats)
    print("\n" + stats_report)

    # Save stats to JSON
    stats_path = Path("outputs/reports/drift_analysis_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved drift statistics: {stats_path}")

    # Find optimal threshold
    logger.info("Finding optimal drift threshold...")
    optimal_threshold, best_f1, threshold_metrics = detector.find_optimal_threshold(
        results["drift_scores"],
        results["labels"],
        safe_label=0,  # SAFE
        advice_labels=[1, 2],  # INVESTMENT_ADVICE, INDIRECT_ADVICE
    )

    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"F1 Score: {threshold_metrics['f1']:.4f}")
    print(f"Accuracy: {threshold_metrics['accuracy']:.4f}")
    print("\nClassification Report (SAFE vs ADVICE):")
    print(
        f"  SAFE     - Precision: {threshold_metrics['classification_report']['SAFE']['precision']:.4f}, "
        f"Recall: {threshold_metrics['classification_report']['SAFE']['recall']:.4f}, "
        f"F1: {threshold_metrics['classification_report']['SAFE']['f1-score']:.4f}"
    )
    print(
        f"  ADVICE   - Precision: {threshold_metrics['classification_report']['ADVICE']['precision']:.4f}, "
        f"Recall: {threshold_metrics['classification_report']['ADVICE']['recall']:.4f}, "
        f"F1: {threshold_metrics['classification_report']['ADVICE']['f1-score']:.4f}"
    )

    # Save threshold metrics
    threshold_path = Path("outputs/reports/drift_threshold_metrics.json")
    with open(threshold_path, "w") as f:
        json.dump(threshold_metrics, f, indent=2)
    logger.info(f"Saved threshold metrics: {threshold_path}")

    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer = DriftVisualizer()

    # 1. Drift distributions (box plot + violin plot)
    visualizer.plot_drift_distributions(
        results["drift_scores"],
        results["labels"],
        class_names,
        output_path="outputs/visualizations/drift_distributions.png",
    )

    # 2. Drift histogram with threshold
    visualizer.plot_drift_histogram(
        results["drift_scores"],
        results["labels"],
        class_names,
        output_path="outputs/visualizations/drift_histogram.png",
        optimal_threshold=optimal_threshold,
    )

    # 3. Mean drift comparison
    visualizer.plot_drift_comparison(
        stats,
        output_path="outputs/visualizations/drift_comparison.png",
    )

    # 4. UMAP with drift (if UMAP projections exist)
    try:
        cache = EmbeddingCache()
        # Try to load existing UMAP projections
        umap_projection_path = Path(
            "outputs/visualizations/sentence_transformer_projections.npy"
        )
        if umap_projection_path.exists():
            logger.info("Loading existing UMAP projections...")
            umap_projections = np.load(umap_projection_path)

            visualizer.plot_umap_with_drift(
                umap_projections,
                results["drift_scores"],
                results["labels"],
                class_names,
                output_path="outputs/visualizations/drift_umap_overlay.png",
            )
        else:
            logger.warning(
                "UMAP projections not found. Run Phase 3 first for UMAP overlay."
            )
    except Exception as e:
        logger.warning(f"Could not create UMAP drift overlay: {e}")

    # Compare to baseline RF performance (if available)
    logger.info("Comparing to Random Forest baseline...")
    rf_metrics_path = Path("outputs/metrics/classification_report.json")

    if rf_metrics_path.exists():
        with open(rf_metrics_path, "r") as f:
            rf_metrics = json.load(f)

        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print(
            f"Random Forest (original):   Accuracy: {rf_metrics['accuracy']:.4f}, "
            f"F1: {rf_metrics['f1_score']:.4f}"
        )
        print(
            f"Drift Detection (SAFE vs ADVICE): Accuracy: {threshold_metrics['accuracy']:.4f}, "
            f"F1: {threshold_metrics['f1']:.4f}"
        )
        print("")

        # Calculate improvement
        improvement = threshold_metrics['accuracy'] - rf_metrics['accuracy']
        if improvement > 0:
            print(
                f"✓ Drift detection shows +{improvement:.2%} accuracy improvement for SAFE vs ADVICE!"
            )
        else:
            print(
                f"✗ Drift detection is {abs(improvement):.2%} lower than RF for this binary task"
            )
            print(
                "  Note: RF was trained on 5-way classification, drift is binary (SAFE vs ADVICE)"
            )
    else:
        logger.warning("RF metrics not found. Run Phase 4 first for comparison.")

    # Analyze hypothesis
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)

    safe_mean = stats["SAFE"]["mean"]
    investment_mean = stats["INVESTMENT_ADVICE"]["mean"]
    indirect_mean = stats["INDIRECT_ADVICE"]["mean"]

    print(f"SAFE mean drift:             {safe_mean:.4f}")
    print(f"INVESTMENT_ADVICE mean drift: {investment_mean:.4f}")
    print(f"INDIRECT_ADVICE mean drift:   {indirect_mean:.4f}")
    print("")

    gap_investment = investment_mean - safe_mean
    gap_indirect = indirect_mean - safe_mean

    print(f"Gap (INVESTMENT vs SAFE):    {gap_investment:.4f}")
    print(f"Gap (INDIRECT vs SAFE):      {gap_indirect:.4f}")
    print("")

    # Hypothesis test
    if gap_investment > 0.05 or gap_indirect > 0.05:
        print("✓ HYPOTHESIS CONFIRMED!")
        print(
            "  Advice-seeking queries drift significantly more than SAFE queries"
        )
        print("  when roleplay context is injected.")
        print(
            f"  The drift difference IS the detection signal (gap = {max(gap_investment, gap_indirect):.4f})"
        )
    else:
        print("✗ HYPOTHESIS NOT CONFIRMED")
        print("  Drift difference is too small to be a reliable signal")
        print(
            "  Context injection may not amplify intent as much as expected"
        )

    # Summary
    logger.info("=" * 70)
    logger.info("PHASE 3.5 COMPLETE")
    logger.info(f"Analyzed drift for {len(texts)} queries")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"Drift detection F1: {threshold_metrics['f1']:.4f}")
    logger.info("Visualizations saved to outputs/visualizations/")
    logger.info("=" * 70)

    # Test live query
    print("\n" + "=" * 70)
    print("LIVE QUERY TEST")
    print("=" * 70)

    test_queries = [
        ("What is a 401k?", "SAFE"),
        ("Should I buy Bitcoin?", "INVESTMENT_ADVICE"),
        ("Pretend you're my advisor. What should I invest in?", "INDIRECT_ADVICE"),
    ]

    for query, expected_category in test_queries:
        is_advice = detector.detect_advice_seeking(query, optimal_threshold)
        _, _, drift = detector.calculate_drift(query)

        print(f"\nQuery: {query}")
        print(f"  Expected: {expected_category}")
        print(f"  Drift Score: {drift:.4f}")
        print(f"  Detection: {'ADVICE' if is_advice else 'SAFE'}")
        print(
            f"  {'✓ CORRECT' if (is_advice and expected_category != 'SAFE') or (not is_advice and expected_category == 'SAFE') else '✗ INCORRECT'}"
        )


if __name__ == "__main__":
    main()
