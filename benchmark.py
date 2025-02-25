import time
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import argparse
from typing import Dict, List, Any, Tuple
import tempfile

# Import necessary Langflow components
# Note: You may need to adjust imports based on Langflow's actual API
from langflow.interface.run import load_flow_from_json, execute_flow
from langflow.utils.logger import logger

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutomatedBenchmarkRunner:
    def __init__(self, 
                 flow_path: str, 
                 iterations: int = 10, 
                 warmup: int = 2,
                 output_dir: str = None,
                 is_optimized: bool = False):
        """
        Initialize benchmark runner for Langflow flow execution.
        
        Args:
            flow_path: Path to the complex flow JSON file
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations (not counted in results)
            output_dir: Directory to save results
            is_optimized: Whether this run is for the optimized version
        """
        self.flow_path = Path(flow_path)
        self.iterations = iterations
        self.warmup = warmup
        self.results = []
        self.is_optimized = is_optimized
        self.version_label = "optimized" if is_optimized else "original"
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="langflow_benchmark_"))
        
        if not self.flow_path.exists():
            raise FileNotFoundError(f"Flow file not found: {flow_path}")
        
        logger.info(f"Loading flow from {self.flow_path}")
        with open(self.flow_path, "r") as f:
            self.flow_data = json.load(f)
    
    def create_complex_test_data(self) -> Dict[str, Any]:
        """Create complex test data for the flow execution."""
        # Customize this based on your flow's input requirements
        return {
            "user_query": "Analyze the performance trends in our quarterly financial reports and suggest areas for improvement",
            "context": "The company has been experiencing fluctuating growth patterns across different product lines. Q2 showed a 15% decline in hardware sales while software subscriptions increased by 23%.",
            "parameters": {
                "depth": "detailed",
                "format": "business_report",
                "include_charts": True,
                "timeframe": "last_4_quarters"
            },
            "historical_data": [
                {"quarter": "Q1", "revenue": 1200000, "costs": 800000},
                {"quarter": "Q2", "revenue": 1350000, "costs": 950000},
                {"quarter": "Q3", "revenue": 1100000, "costs": 820000},
                {"quarter": "Q4", "revenue": 1420000, "costs": 910000}
            ]
        }
    
    def run_benchmark(self):
        """Run benchmark on the flow implementation."""
        logger.info(f"Running benchmark on {self.version_label} implementation...")
        test_data = self.create_complex_test_data()
        
        # Warmup runs
        for i in range(self.warmup):
            logger.info(f"Warmup run {i+1}/{self.warmup}")
            flow = load_flow_from_json(self.flow_data)
            _ = execute_flow(flow, test_data)
        
        # Benchmark runs
        for i in range(self.iterations):
            logger.info(f"Benchmark run {i+1}/{self.iterations}")
            flow = load_flow_from_json(self.flow_data)
            
            start_time = time.time()
            result = execute_flow(flow, test_data)
            execution_time = time.time() - start_time
            
            self.results.append({
                "execution_time": execution_time,
                "output_size": self._measure_output_size(result),
                "iteration": i + 1
            })
            
            logger.info(f"Execution time: {execution_time:.4f}s")
        
        return self.results
    
    def _measure_output_size(self, result: Any) -> int:
        """Measure the size of the output in bytes."""
        # This is a simple approximation - adjust based on your actual result structure
        return len(json.dumps(result)) if result else 0
    
    def generate_statistics(self) -> Dict[str, float]:
        """Generate statistics from benchmark results."""
        if not self.results:
            logger.warning("No benchmark results to generate statistics from")
            return {}
        
        times = [r["execution_time"] for r in self.results]
        stats = {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0
        }
        
        return stats
    
    def save_results(self):
        """Save benchmark results to a JSON file."""
        results_file = self.output_dir / f"{self.version_label}_benchmark_results.json"
        
        results = {
            "version": self.version_label,
            "results": self.results,
            "statistics": self.generate_statistics(),
            "flow_name": self.flow_path.stem,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": self.iterations,
            "warmup_iterations": self.warmup
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        return results_file
    
    def visualize_results(self):
        """Generate visualization of benchmark results."""
        if not self.results:
            logger.warning("No benchmark results to visualize")
            return None
        
        viz_file = self.output_dir / f"{self.version_label}_visualization.png"
        
        plt.figure(figsize=(10, 6))
        
        # Plot execution times
        iterations = range(1, len(self.results) + 1)
        times = [r["execution_time"] for r in self.results]
        
        color = 'green' if self.is_optimized else 'blue'
        plt.plot(iterations, times, f'{color}o-', label=f'{self.version_label.capitalize()} Version')
        
        plt.title(f'Flow Execution Time - {self.version_label.capitalize()} Version')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.legend()
        
        # Add statistics to the plot
        stats = self.generate_statistics()
        stats_text = f"Mean: {stats['mean']:.4f}s\nMedian: {stats['median']:.4f}s\nStdev: {stats['stdev']:.4f}s"
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_file)
        logger.info(f"Visualization saved to {viz_file}")
        
        return viz_file

def compare_results(original_results_path: str, optimized_results_path: str, output_dir: Path):
    """Compare original and optimized benchmark results."""
    # Load results
    with open(original_results_path, 'r') as f:
        original_data = json.load(f)
    
    with open(optimized_results_path, 'r') as f:
        optimized_data = json.load(f)
    
    # Extract statistics
    orig_stats = original_data["statistics"]
    opt_stats = optimized_data["statistics"]
    
    # Calculate improvement
    improvement = ((orig_stats["mean"] - opt_stats["mean"]) / orig_stats["mean"]) * 100
    
    # Generate comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Execution times comparison
    plt.subplot(2, 1, 1)
    
    orig_times = [r["execution_time"] for r in original_data["results"]]
    opt_times = [r["execution_time"] for r in optimized_data["results"]]
    
    max_iters = max(len(orig_times), len(opt_times))
    iterations = range(1, max_iters + 1)
    
    plt.plot(iterations[:len(orig_times)], orig_times, 'bo-', label='Original')
    plt.plot(iterations[:len(opt_times)], opt_times, 'go-', label='Artemis Optimized')
    
    plt.title('Execution Time Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Bar chart comparison
    plt.subplot(2, 1, 2)
    
    labels = ['Original', 'Artemis Optimized']
    means = [orig_stats["mean"], opt_stats["mean"]]
    errors = [orig_stats["stdev"], opt_stats["stdev"]]
    
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, color=['blue', 'green'], alpha=0.7, capsize=10)
    plt.xticks(x, labels)
    plt.title('Average Execution Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y')
    
    # Add improvement text
    plt.text(1, means[1] + errors[1] + 0.05, 
            f"{improvement:.1f}% improvement", 
            ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_file = output_dir / "benchmark_comparison.png"
    plt.savefig(comparison_file)
    
    # Create comparison results JSON
    comparison = {
        "original": orig_stats,
        "optimized": opt_stats,
        "improvement_percentage": improvement,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    comparison_json = output_dir / "benchmark_comparison.json"
    with open(comparison_json, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison visualization saved to {comparison_file}")
    logger.info(f"Comparison data saved to {comparison_json}")
    
    return comparison

def create_complex_flow(output_path: str):
    """Create a complex flow for testing."""
    # This function would create a complex flow with multiple branches
    # For demo purposes, let's create a structure that represents a complex flow
    
    # This is a placeholder - replace with actual Langflow format
    flow = {
        "nodes": [
            {"id": "input", "type": "InputNode", "data": {"name": "Input"}},
            {"id": "text_splitter", "type": "TextSplitterNode", "data": {"chunk_size": 1000}},
            {"id": "embeddings", "type": "EmbeddingsNode", "data": {"model": "text-embedding-ada-002"}},
            {"id": "vector_store", "type": "VectorStoreNode", "data": {"type": "FAISS"}},
            {"id": "retriever", "type": "RetrieverNode", "data": {"k": 4}},
            {"id": "router", "type": "RouterNode", "data": {"routing_field": "query_type"}},
            {"id": "summarize_prompt", "type": "PromptNode", "data": {"template": "Summarize the following: {context}"}},
            {"id": "analyze_prompt", "type": "PromptNode", "data": {"template": "Analyze the following data: {context}"}},
            {"id": "forecast_prompt", "type": "PromptNode", "data": {"template": "Forecast based on: {context}"}},
            {"id": "llm_gpt4", "type": "LLMNode", "data": {"model": "gpt-4", "temperature": 0.7}},
            {"id": "llm_claude", "type": "LLMNode", "data": {"model": "claude-3-opus-20240229", "temperature": 0.5}},
            {"id": "memory", "type": "MemoryNode", "data": {"type": "buffer", "k": 5}},
            {"id": "formatter", "type": "FormatterNode", "data": {"format": "markdown"}},
            {"id": "output", "type": "OutputNode", "data": {"name": "Final Response"}}
        ],
        "edges": [
            {"source": "input", "target": "text_splitter"},
            {"source": "text_splitter", "target": "embeddings"},
            {"source": "embeddings", "target": "vector_store"},
            {"source": "vector_store", "target": "retriever"},
            {"source": "input", "target": "router"},
            {"source": "router", "sourceHandle": "summarize", "target": "summarize_prompt"},
            {"source": "router", "sourceHandle": "analyze", "target": "analyze_prompt"},
            {"source": "router", "sourceHandle": "forecast", "target": "forecast_prompt"},
            {"source": "retriever", "target": "summarize_prompt"},
            {"source": "retriever", "target": "analyze_prompt"},
            {"source": "retriever", "target": "forecast_prompt"},
            {"source": "summarize_prompt", "target": "llm_gpt4"},
            {"source": "analyze_prompt", "target": "llm_claude"},
            {"source": "forecast_prompt", "target": "llm_gpt4"},
            {"source": "llm_gpt4", "target": "memory"},
            {"source": "llm_claude", "target": "memory"},
            {"source": "memory", "target": "formatter"},
            {"source": "formatter", "target": "output"}
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(flow, f, indent=2)
    
    logger.info(f"Created complex flow at {output_path}")
    return output_path

def main():
    """Main function to run the automated benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark Langflow execution performance')
    parser.add_argument('--flow', type=str, help='Path to existing flow JSON file')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup iterations')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--mode', choices=['original', 'optimized', 'both'], default='both', 
                        help='Which version to benchmark')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create or use existing flow
    flow_path = args.flow
    if not flow_path or not Path(flow_path).exists():
        flow_path = create_complex_flow(str(output_dir / "complex_flow.json"))
    
    # Run benchmarks
    original_results_path = None
    optimized_results_path = None
    
    if args.mode in ['original', 'both']:
        logger.info("Running benchmark on original implementation")
        original_benchmark = AutomatedBenchmarkRunner(
            flow_path=flow_path,
            iterations=args.iterations,
            warmup=args.warmup,
            output_dir=str(output_dir),
            is_optimized=False
        )
        original_benchmark.run_benchmark()
        original_results_path = original_benchmark.save_results()
        original_benchmark.visualize_results()
    
    if args.mode in ['optimized', 'both']:
        logger.info("Running benchmark on optimized implementation")
        optimized_benchmark = AutomatedBenchmarkRunner(
            flow_path=flow_path,
            iterations=args.iterations,
            warmup=args.warmup,
            output_dir=str(output_dir),
            is_optimized=True
        )
        optimized_benchmark.run_benchmark()
        optimized_results_path = optimized_benchmark.save_results()
        optimized_benchmark.visualize_results()
    
    # Generate comparison if both versions were benchmarked
    if args.mode == 'both' and original_results_path and optimized_results_path:
        comparison = compare_results(original_results_path, optimized_results_path, output_dir)
        
        # Display improvement summary
        improvement = comparison["improvement_percentage"]
        logger.info("=" * 50)
        logger.info(f"Performance Improvement: {improvement:.2f}%")
        logger.info(f"Original mean execution time: {comparison['original']['mean']:.4f}s")
        logger.info(f"Optimized mean execution time: {comparison['optimized']['mean']:.4f}s")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()