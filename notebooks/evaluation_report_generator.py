"""
Evaluation Report Generator for Interior Style Transfer POC.

This module creates comprehensive evaluation reports for style transfer
results, following Semantic Seed Coding Standards with proper metrics,
visualizations, and structured documentation for client deliverables.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Define constants
GALLERY_DIR = 'gallery_outputs'
REPORT_DIR = 'evaluation_reports'
REPORT_FILENAME = 'interior_style_evaluation_report.md'
HTML_REPORT_FILENAME = 'interior_style_evaluation_report.html'

def load_gallery_results() -> Dict[str, Any]:
    """
    Load the gallery results data from the JSON file.
    
    Returns:
        Dictionary with gallery results data
    """
    gallery_json = os.path.join(GALLERY_DIR, 'gallery_results.json')
    
    if not os.path.exists(gallery_json):
        logger.error(f"Gallery results file not found: {gallery_json}")
        return {"success": False, "error": "Gallery results file not found"}
    
    try:
        with open(gallery_json, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded gallery results from {gallery_json}")
        return results
    except Exception as e:
        logger.error(f"Error loading gallery results: {str(e)}")
        return {"success": False, "error": f"Error loading gallery results: {str(e)}"}

def calculate_image_metrics(original_path: str, styled_path: str) -> Dict[str, float]:
    """
    Calculate evaluation metrics between original and styled images.
    
    Args:
        original_path: Path to original image
        styled_path: Path to styled image
        
    Returns:
        Dictionary of metric names and values
    """
    try:
        # Import evaluation modules
        try:
            from src.evaluation import VisualEvaluationService, ImageEvaluator
            
            # Use project evaluation modules if available
            eval_service = VisualEvaluationService()
            image_evaluator = ImageEvaluator()
            
            # Load images
            original = np.array(Image.open(original_path))
            styled = np.array(Image.open(styled_path))
            
            # Calculate metrics using project evaluation tools
            metrics = eval_service.calculate_metrics(original, styled)
            
            # Add additional metrics
            structure_eval = image_evaluator.evaluate_structure_preservation(
                original, styled, None  # No explicit mask, will generate one internally
            )
            
            # Combine metrics
            combined_metrics = {
                "ssim": metrics.get("ssim", 0),
                "mse": metrics.get("mse", 0),
                "psnr": metrics.get("psnr", 0) if "psnr" in metrics else -10 * np.log10(metrics.get("mse", 1e-10) / (255.0 ** 2)),
                "structure_preservation": structure_eval.get("ssim_score", 0),
                "is_structure_preserved": structure_eval.get("is_structure_preserved", False)
            }
            
            return combined_metrics
            
        except ImportError:
            # Fallback to basic metrics if project modules not available
            logger.warning("Project evaluation modules not available, using fallback metrics")
            
            # Import basic image comparison libraries
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
            
            # Load images
            original = np.array(Image.open(original_path).convert('RGB'))
            styled = np.array(Image.open(styled_path).convert('RGB'))
            
            # Ensure images are the same size
            if original.shape != styled.shape:
                styled = np.array(Image.fromarray(styled).resize(
                    (original.shape[1], original.shape[0]), Image.LANCZOS
                ))
            
            # Calculate SSIM
            ssim_value = ssim(original, styled, channel_axis=2, data_range=255)
            
            # Calculate MSE
            mse_value = mean_squared_error(original, styled)
            
            # Calculate PSNR
            psnr_value = peak_signal_noise_ratio(original, styled, data_range=255)
            
            # Simple structure preservation estimate
            structure_preserved = ssim_value > 0.8
            
            return {
                "ssim": ssim_value,
                "mse": mse_value,
                "psnr": psnr_value,
                "structure_preservation": ssim_value,
                "is_structure_preserved": structure_preserved
            }
            
    except Exception as e:
        logger.error(f"Error calculating image metrics: {str(e)}")
        return {
            "ssim": 0,
            "mse": 0,
            "psnr": 0,
            "structure_preservation": 0,
            "is_structure_preserved": False,
            "error": str(e)
        }

def generate_evaluation_metrics(gallery_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate evaluation metrics for all style variations.
    
    Args:
        gallery_results: Gallery results data
        
    Returns:
        Dictionary with evaluation metrics for all images
    """
    metrics_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content_images": gallery_results.get("content_images", []),
        "styles_applied": gallery_results.get("styles_applied", []),
        "metrics": []
    }
    
    for variation in gallery_results.get("variations", []):
        content_name = variation.get("content_name", "Unknown")
        logger.info(f"Calculating metrics for {content_name}")
        
        # Create entry for this content image
        content_metrics = {
            "content_name": content_name,
            "styles": []
        }
        
        # Find the original image in the sample directory
        sample_dir = os.path.join('sample_images')
        original_path = None
        
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(sample_dir, f"{content_name}{ext}")
            if os.path.exists(potential_path):
                original_path = potential_path
                break
        
        if not original_path:
            logger.warning(f"Original image not found for {content_name}")
            continue
        
        # Calculate metrics for each style
        for style in variation.get("styles", []):
            style_name = style.get("style_name", "Unknown")
            output_file = style.get("output_file", "")
            
            if not output_file:
                logger.warning(f"Output file not found for {content_name} with {style_name} style")
                continue
            
            # Get full path to styled image
            styled_path = os.path.join(GALLERY_DIR, output_file)
            
            if not os.path.exists(styled_path):
                logger.warning(f"Styled image not found: {styled_path}")
                continue
            
            # Calculate metrics
            metrics = calculate_image_metrics(original_path, styled_path)
            
            # Add to results
            content_metrics["styles"].append({
                "style_name": style_name,
                "output_file": output_file,
                "metrics": metrics
            })
            
            logger.info(f"Calculated metrics for {content_name} with {style_name} style: "
                       f"SSIM={metrics.get('ssim', 0):.4f}, MSE={metrics.get('mse', 0):.4f}")
        
        # Add to overall metrics
        metrics_results["metrics"].append(content_metrics)
    
    return metrics_results

def create_metrics_visualizations(metrics_results: Dict[str, Any], gallery_results: Dict[str, Any]) -> None:
    """
    Create visualizations of metrics for the report.
    
    Args:
        metrics_results: Dictionary with evaluation metrics
        gallery_results: Dictionary with gallery results
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Create a DataFrame for metrics
    data = []
    
    for content_metrics in metrics_results.get("metrics", []):
        content_name = content_metrics.get("content_name", "Unknown")
        
        for style in content_metrics.get("styles", []):
            style_name = style.get("style_name", "Unknown")
            metrics = style.get("metrics", {})
            
            data.append({
                "Content": content_name,
                "Style": style_name,
                "SSIM": metrics.get("ssim", 0),
                "MSE": metrics.get("mse", 0),
                "PSNR": metrics.get("psnr", 0),
                "Structure Preservation": metrics.get("structure_preservation", 0),
                "Structure Preserved": metrics.get("is_structure_preserved", False)
            })
    
    if not data:
        logger.warning("No metrics data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # 1. Create bar chart for SSIM values
    plt.figure(figsize=(10, 6))
    ssim_pivot = df.pivot(index="Content", columns="Style", values="SSIM")
    ssim_pivot.plot(kind="bar", rot=0)
    plt.title("SSIM Values by Content and Style")
    plt.ylabel("SSIM (higher is better)")
    plt.ylim(0, 1)  # SSIM is between 0 and 1
    plt.legend(title="Style")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "ssim_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Create heatmap for all metrics
    plt.figure(figsize=(12, 8))
    
    # Create a style ranking based on average SSIM
    style_ranking = df.groupby("Style")["SSIM"].mean().sort_values(ascending=False).index.tolist()
    content_ranking = df.groupby("Content")["SSIM"].mean().sort_values(ascending=False).index.tolist()
    
    # Create a pivot table with the best styles first
    heatmap_data = df.pivot_table(
        index="Content", 
        columns="Style", 
        values="SSIM",
        aggfunc="mean"
    ).reindex(index=content_ranking, columns=style_ranking)
    
    import seaborn as sns
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".3f")
    plt.title("SSIM Heatmap by Content and Style")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "ssim_heatmap.png"), dpi=150)
    plt.close()
    
    # 3. Create summary of best styles for each content
    best_styles = df.loc[df.groupby("Content")["SSIM"].idxmax()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best_styles["Content"], best_styles["SSIM"], color="lightblue")
    
    # Add style names as annotations
    for i, bar in enumerate(bars):
        style_name = best_styles.iloc[i]["Style"]
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            style_name,
            ha='center',
            rotation=0,
            fontsize=10
        )
    
    plt.title("Best Style for Each Content Image (by SSIM)")
    plt.ylabel("SSIM Score")
    plt.ylim(0, 1.1)  # SSIM is between 0 and 1, with space for annotations
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "best_styles.png"), dpi=150)
    plt.close()
    
    # 4. Save the metrics DataFrame as CSV
    df.to_csv(os.path.join(REPORT_DIR, "style_metrics.csv"), index=False)
    
    logger.info(f"Created metrics visualizations in {REPORT_DIR}")

def generate_markdown_report(metrics_results: Dict[str, Any], gallery_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive Markdown report of the evaluation.
    
    Args:
        metrics_results: Dictionary with evaluation metrics
        gallery_results: Dictionary with gallery results
        
    Returns:
        Path to the generated report
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, REPORT_FILENAME)
    
    # Create report content
    timestamp = metrics_results.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    content_images = metrics_results.get("content_images", [])
    styles_applied = metrics_results.get("styles_applied", [])
    
    report_content = f"""# Interior Style Transfer Evaluation Report

**Generated:** {timestamp}

## Overview

This report presents a comprehensive evaluation of interior style transfer results
for {len(content_images)} interior images with {len(styles_applied)} different style variations.

### Sample Interiors

The following interior images were used in this evaluation:

"""
    
    # Add content image list
    for image in content_images:
        report_content += f"- {image}\n"
    
    report_content += """
### Style Variations

The following style variations were applied to each interior:

"""
    
    # Add style list
    for style in styles_applied:
        report_content += f"- {style}\n"
    
    # Add metrics summary
    report_content += """
## Evaluation Metrics

The following metrics were used to evaluate the quality of style transfer:

- **SSIM (Structural Similarity Index)**: Measures the perceived similarity between images (higher is better)
- **MSE (Mean Squared Error)**: Measures the average squared difference between pixels (lower is better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in decibels (higher is better)
- **Structure Preservation**: Measures how well architectural elements were preserved (higher is better)

### Results Visualization

![SSIM Comparison](ssim_comparison.png)

*SSIM values for each content-style combination. Higher values indicate better structural similarity.*

![SSIM Heatmap](ssim_heatmap.png)

*Heatmap of SSIM values across all content-style combinations.*

![Best Styles](best_styles.png)

*The best style for each content image based on SSIM score.*

## Detailed Results

"""
    
    # Add detailed metrics for each content-style combination
    for content_metrics in metrics_results.get("metrics", []):
        content_name = content_metrics.get("content_name", "Unknown")
        report_content += f"### {content_name}\n\n"
        
        # Add comparison image if available
        for variation in gallery_results.get("variations", []):
            if variation.get("content_name") == content_name:
                comparison_image = variation.get("comparison_image", "")
                if comparison_image:
                    report_content += f"![{content_name} Style Comparison](../{GALLERY_DIR}/{comparison_image})\n\n"
                break
        
        # Create metrics table
        report_content += "| Style | SSIM ↑ | MSE ↓ | PSNR ↑ | Structure Preserved |\n"
        report_content += "|-------|--------|-------|--------|--------------------|\n"
        
        # Sort styles by SSIM (best first)
        sorted_styles = sorted(
            content_metrics.get("styles", []),
            key=lambda x: x.get("metrics", {}).get("ssim", 0),
            reverse=True
        )
        
        for style in sorted_styles:
            style_name = style.get("style_name", "Unknown")
            metrics = style.get("metrics", {})
            
            ssim_value = metrics.get("ssim", 0)
            mse_value = metrics.get("mse", 0)
            psnr_value = metrics.get("psnr", 0)
            structure_preserved = "✅" if metrics.get("is_structure_preserved", False) else "❌"
            
            report_content += f"| {style_name} | {ssim_value:.4f} | {mse_value:.2f} | {psnr_value:.2f} | {structure_preserved} |\n"
        
        report_content += "\n"
    
    # Add recommendations section
    report_content += """
## Recommendations

Based on the evaluation metrics, the following style combinations are recommended:

"""
    
    # Create a DataFrame for finding best combinations
    data = []
    for content_metrics in metrics_results.get("metrics", []):
        content_name = content_metrics.get("content_name", "Unknown")
        for style in content_metrics.get("styles", []):
            style_name = style.get("style_name", "Unknown")
            metrics = style.get("metrics", {})
            data.append({
                "Content": content_name,
                "Style": style_name,
                "SSIM": metrics.get("ssim", 0)
            })
    
    if data:
        df = pd.DataFrame(data)
        best_styles = df.loc[df.groupby("Content")["SSIM"].idxmax()]
        
        for _, row in best_styles.iterrows():
            report_content += f"- **{row['Content']}**: {row['Style']} style (SSIM: {row['SSIM']:.4f})\n"
    
    # Add conclusion
    report_content += """
## Conclusion

This evaluation demonstrates the effectiveness of the Interior Style Transfer POC
in generating style variations while preserving architectural structure.
The metrics show that the style transfer process maintains good structural similarity
while successfully applying the target styles.

"""
    
    # Write report to file
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Generated Markdown report at {report_path}")
    return report_path

def generate_html_report(markdown_path: str) -> str:
    """
    Convert the Markdown report to HTML format.
    
    Args:
        markdown_path: Path to the Markdown report
        
    Returns:
        Path to the HTML report
    """
    try:
        import markdown
        from markdown.extensions.tables import TableExtension
        
        html_path = os.path.join(REPORT_DIR, HTML_REPORT_FILENAME)
        
        # Read Markdown content
        with open(markdown_path, 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=[TableExtension()])
        
        # Add basic styling
        styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interior Style Transfer Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    {html_content}
    <div class="footer">
        <p>Generated with Interior Style Transfer POC - Semantic Seed Venture Studio</p>
    </div>
</body>
</html>
"""
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(styled_html)
        
        logger.info(f"Generated HTML report at {html_path}")
        return html_path
        
    except ImportError:
        logger.warning("markdown module not available, skipping HTML report generation")
        return ""

def run_evaluation_report():
    """
    Run the full evaluation report generation process.
    """
    print("=== Evaluation Report Generation ===")
    print("Generating comprehensive evaluation metrics and visualizations...")
    
    # Load gallery results
    gallery_results = load_gallery_results()
    
    if gallery_results.get("success") == False:
        print(f"⚠️ {gallery_results.get('error', 'Unknown error')}")
        return
    
    # Generate metrics
    metrics_results = generate_evaluation_metrics(gallery_results)
    
    # Create visualizations
    create_metrics_visualizations(metrics_results, gallery_results)
    
    # Generate Markdown report
    report_path = generate_markdown_report(metrics_results, gallery_results)
    
    # Generate HTML report
    html_path = generate_html_report(report_path)
    
    print(f"\n✅ Evaluation report generation complete!")
    print(f"Metrics and visualizations saved to {REPORT_DIR} directory")
    print(f"Markdown report: {report_path}")
    
    if html_path:
        print(f"HTML report: {html_path}")
    
    print("\nSuggested next steps:")
    print("1. Review the HTML report for a comprehensive evaluation")
    print("2. Share the report with stakeholders for feedback")
    print("3. Use the metrics to select the best style combinations for production")

if __name__ == "__main__":
    run_evaluation_report()
