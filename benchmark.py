#!/usr/bin/env python3
import os
import sys
import json
import time
import traceback
import statistics
from pathlib import Path
import argparse
from datetime import datetime

# Add the necessary paths for importing langflow modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src/backend/base"))

try:
    from langflow.logging.logger import logger
    from langflow.load.load import load_flow_from_json
except ImportError as e:
    print(f"Error importing Langflow modules: {e}")
    print("Please ensure the Langflow package is properly installed")
    sys.exit(1)

class LangflowBenchmark:
    def __init__(self, verbose=True):
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verbose = verbose
        
        # Flow example paths - adjust to your repository
        self.flow_paths = [
            "docs/docs/Integrations/AssemblyAI_Flow.json",
            "docs/docs/Integrations/Notion/Conversational_Notion_Agent.json"
        ]
        
        # Check if files exist
        self.valid_paths = []
        for path in self.flow_paths:
            if os.path.exists(path):
                self.valid_paths.append(path)
            else:
                print(f"Warning: Flow file not found: {path}")
                
        if not self.valid_paths:
            print("No valid flow files found!")
    
    def log(self, message):
        """Custom logging function"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def benchmark_flow_loading(self, flow_path, iterations=3, warmup=1):
        """Benchmark the time it takes to load a flow."""
        flow_name = os.path.basename(flow_path).replace('.json', '')
        self.log(f"Benchmarking loading time for {flow_name}")
        
        # Load the flow data
        try:
            with open(flow_path, 'r') as f:
                flow_data = json.load(f)
                self.log(f"Successfully loaded flow file: {flow_path}")
                if self.verbose:
                    self.log(f"Flow structure: {list(flow_data.keys())}")
                    if "data" in flow_data and "nodes" in flow_data.get("data", {}):
                        self.log(f"Number of nodes: {len(flow_data['data']['nodes'])}")
        except Exception as e:
            self.log(f"Error loading flow file {flow_path}: {str(e)}")
            return {
                "name": flow_name,
                "path": flow_path,
                "status": "error",
                "error": str(e)
            }
        
        # Warmup runs
        for i in range(warmup):
            self.log(f"Warmup run {i+1}/{warmup}")
            try:
                _ = load_flow_from_json(flow_data)
                self.log(f"Warmup run {i+1} completed successfully")
            except Exception as e:
                self.log(f"Error during warmup run {i+1}: {str(e)}")
                if self.verbose:
                    self.log(traceback.format_exc())
                return {
                    "name": flow_name,
                    "path": flow_path,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc() if self.verbose else None
                }
        
        # Timed runs
        load_times = []
        for i in range(iterations):
            self.log(f"Benchmark run {i+1}/{iterations}")
            start = time.time()
            try:
                graph = load_flow_from_json(flow_data)
                end = time.time()
                load_time = end - start
                load_times.append(load_time)
                self.log(f"Run {i+1} completed in {load_time:.4f}s")
            except Exception as e:
                self.log(f"Error during benchmark run {i+1}: {str(e)}")
                if self.verbose:
                    self.log(traceback.format_exc())
                return {
                    "name": flow_name,
                    "path": flow_path,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc() if self.verbose else None
                }
        
        if not load_times:
            self.log(f"No successful runs for {flow_name}")
            return {
                "name": flow_name,
                "path": flow_path,
                "status": "error",
                "error": "No successful benchmark runs"
            }
        
        # Calculate statistics
        avg_time = statistics.mean(load_times)
        median_time = statistics.median(load_times)
        min_time = min(load_times)
        max_time = max(load_times)
        
        self.log(f"{flow_name} benchmark results:")
        self.log(f"Average: {avg_time:.4f}s")
        self.log(f"Median: {median_time:.4f}s")
        self.log(f"Min: {min_time:.4f}s")
        self.log(f"Max: {max_time:.4f}s")
        
        result = {
            "name": flow_name,
            "path": flow_path,
            "status": "success",
            "type": "load",
            "iterations": iterations,
            "avg_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "all_times": load_times
        }
        
        # Save individual result
        result_file = self.results_dir / f"{flow_name}_result_{self.timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        self.log(f"Results saved to {result_file}")
        
        return result
    
    def run_benchmarks(self, iterations=3, warmup=1):
        """Run benchmarks on all example flows."""
        if not self.valid_paths:
            self.log("No valid flow files to benchmark")
            return []
            
        results = []
        
        for flow_path in self.valid_paths:
            result = self.benchmark_flow_loading(flow_path, iterations, warmup)
            results.append(result)
        
        # Save results
        results_file = self.results_dir / f"benchmark_results_{self.timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.log(f"All benchmark results saved to {results_file}")
        
        # Print summary
        self.log("\nBenchmark Summary:")
        for result in results:
            if result["status"] == "success":
                self.log(f"{result['name']}: Avg {result['avg_time']:.4f}s, Median {result['median_time']:.4f}s")
            else:
                self.log(f"{result['name']}: Error - {result['error']}")
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Langflow Benchmark Tool")
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=3, 
        help="Number of benchmark iterations to run"
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=1, 
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--example", 
        type=str,
        help="Specify a single example flow file to benchmark"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    benchmark = LangflowBenchmark(verbose=args.verbose)
    
    if args.example:
        if os.path.exists(args.example):
            benchmark.log(f"Running benchmark on specified example: {args.example}")
            result = benchmark.benchmark_flow_loading(args.example, args.iterations, args.warmup)
            benchmark.log("\nBenchmark Summary:")
            if result["status"] == "success":
                benchmark.log(f"{result['name']}: Avg {result['avg_time']:.4f}s, Median {result['median_time']:.4f}s")
            else:
                benchmark.log(f"{result['name']}: Error - {result['error']}")
        else:
            benchmark.log(f"Specified example file not found: {args.example}")
    else:
        benchmark.run_benchmarks(iterations=args.iterations, warmup=args.warmup)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())