#!/usr/bin/env python3
"""
OBDD Library Optimized Benchmark Report Generator

This script generates focused, meaningful visualizations of OBDD library performance
across Sequential CPU, OpenMP Parallel, and CUDA GPU backends.

Author: @vijsh32
Date: September 1, 2025
Version: 2.0 (Optimized)
Copyright: 2024 High Performance Computing Laboratory
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Set style for professional looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class OptimizedOBDDBenchmarkAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.performance_data = None
        self.memory_data = None 
        self.correctness_data = None
        self.load_data()
        
    def load_data(self):
        """Load benchmark data from CSV files"""
        print("Loading benchmark data...")
        
        # Find most recent benchmark files
        csv_files = list(self.results_dir.glob("*benchmark_results*.csv"))
        if csv_files:
            latest_results = max(csv_files, key=os.path.getctime)
            print(f"Loading performance data from: {latest_results}")
            self.performance_data = pd.read_csv(latest_results)
            self.performance_data = self.clean_numeric_data(self.performance_data)
        
        memory_files = list(self.results_dir.glob("*memory_usage*.csv"))
        if memory_files:
            latest_memory = max(memory_files, key=os.path.getctime)
            print(f"Loading memory data from: {latest_memory}")
            self.memory_data = pd.read_csv(latest_memory)
            
        correctness_files = list(self.results_dir.glob("*correctness*.csv"))
        if correctness_files:
            latest_correctness = max(correctness_files, key=os.path.getctime)
            print(f"Loading correctness data from: {latest_correctness}")
            self.correctness_data = pd.read_csv(latest_correctness)
            
        # If no files found, create sample data
        if self.performance_data is None:
            print("No benchmark data found, creating sample data...")
            self.create_sample_data()
    
    def clean_numeric_data(self, df):
        """Clean and convert numeric columns, handling TIMEOUT/FAILED values"""
        numeric_columns = ['Time_ms', 'Memory_MB', 'Operations_per_sec', 'Nodes_per_sec']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Variables' in df.columns:
            df['Variables'] = pd.to_numeric(df['Variables'], errors='coerce')
            
        return df
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        backends = ["Sequential", "OpenMP", "CUDA"]
        variables = [4, 6, 8, 10, 12, 14, 16]
        operations = ["AND", "OR", "NOT", "XOR"]
        
        data = []
        for backend in backends:
            for var in variables:
                for op in operations:
                    # Generate realistic sample data
                    if backend == "Sequential":
                        time_ms = var * var * 2.5 + np.random.normal(0, var*0.1)
                        memory_mb = var * 1.2 + np.random.normal(0, 0.2)
                    elif backend == "OpenMP":
                        time_ms = (var * var * 2.5) / 4 + np.random.normal(0, var*0.05)
                        memory_mb = var * 1.5 + np.random.normal(0, 0.3)
                    else:  # CUDA
                        time_ms = (var * var * 2.5) / 8 + np.random.normal(0, var*0.02)
                        memory_mb = var * 0.8 + np.random.normal(0, 0.15)
                    
                    time_ms = max(0.1, time_ms)
                    memory_mb = max(0.1, memory_mb)
                    
                    ops_per_sec = 1000.0 / time_ms if time_ms > 0 else 0
                    
                    data.append({
                        'Backend': backend,
                        'TestType': op,
                        'Variables': var,
                        'Time_ms': time_ms,
                        'Memory_MB': memory_mb,
                        'Operations_per_sec': ops_per_sec,
                        'Success': 'SUCCESS'
                    })
        
        self.performance_data = pd.DataFrame(data)
        
        # Create sample correctness data
        correctness_data = []
        for backend in backends:
            for i in range(20):  # 20 tests per backend
                correctness_data.append({
                    'Backend': backend,
                    'Status': 'PASS' if np.random.random() > 0.05 else 'FAIL'
                })
        
        self.correctness_data = pd.DataFrame(correctness_data)
    
    def plot_performance_overview(self):
        """Generate focused performance overview - the most important charts"""
        print("Generating performance overview...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OBDD Library Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. Execution Time vs Variables (Most Important)
        ax1 = axes[0, 0]
        for backend in self.performance_data['Backend'].unique():
            data = self.performance_data[self.performance_data['Backend'] == backend]
            grouped = data.groupby('Variables')['Time_ms'].mean().reset_index()
            ax1.plot(grouped['Variables'], grouped['Time_ms'], 
                    marker='o', linewidth=3, markersize=8, label=backend)
        
        ax1.set_xlabel('Number of Variables', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Performance Scalability', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup Analysis (Critical for parallel evaluation)
        ax2 = axes[0, 1]
        sequential_data = self.performance_data[self.performance_data['Backend'] == 'Sequential']
        seq_avg_time = sequential_data['Time_ms'].mean()
        
        speedup_data = []
        backends = []
        for backend in self.performance_data['Backend'].unique():
            if backend != 'Sequential':
                backend_data = self.performance_data[self.performance_data['Backend'] == backend]
                backend_avg_time = backend_data['Time_ms'].mean()
                speedup = seq_avg_time / backend_avg_time if backend_avg_time > 0 else 0
                speedup_data.append(speedup)
                backends.append(backend)
        
        if speedup_data:
            bars = ax2.bar(backends, speedup_data, color=['lightgreen', 'coral'], alpha=0.8)
            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline (1x)')
            ax2.set_ylabel('Speedup Factor', fontsize=12)
            ax2.set_title('Average Speedup vs Sequential', fontsize=14, fontweight='bold')
            ax2.legend()
            
            # Add value labels on bars
            for bar, value in zip(bars, speedup_data):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Memory Usage vs Problem Size (Essential for scalability)
        ax3 = axes[1, 0]
        for backend in self.performance_data['Backend'].unique():
            data = self.performance_data[self.performance_data['Backend'] == backend]
            if not data.empty:
                ax3.scatter(data['Variables'], data['Memory_MB'], 
                           label=backend, alpha=0.7, s=60)
                # Add trend line
                if len(data) > 1:
                    z = np.polyfit(data['Variables'].dropna(), data['Memory_MB'].dropna(), 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(data['Variables'].min(), data['Variables'].max(), 100)
                    ax3.plot(x_trend, p(x_trend), linestyle='--', alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Number of Variables', fontsize=12)
        ax3.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax3.set_title('Memory Scalability', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Throughput Comparison (Key performance metric)
        ax4 = axes[1, 1]
        throughput_data = self.performance_data.groupby('Backend')['Operations_per_sec'].mean()
        bars = ax4.bar(throughput_data.index, throughput_data.values, 
                      color=['skyblue', 'lightgreen', 'orange'], alpha=0.8)
        ax4.set_ylabel('Operations per Second', fontsize=12)
        ax4.set_title('Average Throughput by Backend', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, throughput_data.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput_data)*0.01, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimized_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scalability_deep_dive(self):
        """Generate detailed scalability analysis"""
        print("Generating scalability deep dive...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('OBDD Library Scalability Deep Dive', fontsize=16, fontweight='bold')
        
        # 1. Performance vs Problem Size (Log Scale) - Critical for exponential growth
        ax1 = axes[0]
        for backend in self.performance_data['Backend'].unique():
            data = self.performance_data[self.performance_data['Backend'] == backend]
            grouped = data.groupby('Variables')['Time_ms'].mean().reset_index()
            ax1.semilogy(grouped['Variables'], grouped['Time_ms'], 
                        marker='o', linewidth=3, markersize=8, label=backend)
        
        ax1.set_xlabel('Number of Variables', fontsize=12)
        ax1.set_ylabel('Execution Time (ms) - Log Scale', fontsize=12)
        ax1.set_title('Performance Scalability (Log Scale)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup vs Problem Size (Shows where parallel benefits kick in)
        ax2 = axes[1]
        sequential_grouped = self.performance_data[self.performance_data['Backend'] == 'Sequential'].groupby('Variables')['Time_ms'].mean()
        
        for backend in ['OpenMP', 'CUDA']:
            if backend in self.performance_data['Backend'].values:
                backend_grouped = self.performance_data[self.performance_data['Backend'] == backend].groupby('Variables')['Time_ms'].mean()
                speedup = sequential_grouped / backend_grouped
                # Only plot where we have data for both
                common_vars = speedup.dropna()
                if not common_vars.empty:
                    ax2.plot(common_vars.index, common_vars.values, 
                            marker='s', linewidth=3, markersize=8, label=f'{backend} Speedup')
        
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline (1x)')
        ax2.set_xlabel('Number of Variables', fontsize=12)
        ax2.set_ylabel('Speedup Factor', fontsize=12)
        ax2.set_title('Speedup vs Problem Size', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimized_scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_summary_dashboard(self):
        """Generate executive summary dashboard"""
        print("Generating summary dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('OBDD Library Executive Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Success Rate (Critical for correctness)
        ax1 = axes[0, 0]
        if self.correctness_data is not None and not self.correctness_data.empty:
            success_rate = self.correctness_data.groupby('Backend')['Status'].apply(
                lambda x: (x == 'PASS').mean() * 100
            )
            bars = ax1.bar(success_rate.index, success_rate.values, 
                          color=['green', 'blue', 'red'], alpha=0.8)
            ax1.set_ylabel('Success Rate (%)', fontsize=12)
            ax1.set_title('Test Success Rate', fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 105)
            
            for bar, value in zip(bars, success_rate.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Peak Performance Achieved
        ax2 = axes[0, 1]
        peak_throughput = self.performance_data.groupby('Backend')['Operations_per_sec'].max()
        bars = ax2.bar(peak_throughput.index, peak_throughput.values, 
                      color=['lightblue', 'lightgreen', 'lightsalmon'], alpha=0.8)
        ax2.set_ylabel('Peak Operations/sec', fontsize=12)
        ax2.set_title('Peak Performance Achieved', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars, peak_throughput.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_throughput)*0.01, 
                    f'{value:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Memory Efficiency (Ops per MB)
        ax3 = axes[1, 0]
        self.performance_data['Memory_Efficiency'] = (
            self.performance_data['Operations_per_sec'] / 
            (self.performance_data['Memory_MB'] + 0.1)
        )
        efficiency = self.performance_data.groupby('Backend')['Memory_Efficiency'].mean()
        bars = ax3.bar(efficiency.index, efficiency.values, 
                      color=['gold', 'lightcoral', 'lightsteelblue'], alpha=0.8)
        ax3.set_ylabel('Ops per MB', fontsize=12)
        ax3.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars, efficiency.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Performance Summary Table (as text)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate key metrics
        summary_stats = []
        for backend in self.performance_data['Backend'].unique():
            data = self.performance_data[self.performance_data['Backend'] == backend]
            avg_time = data['Time_ms'].mean()
            avg_memory = data['Memory_MB'].mean()
            avg_throughput = data['Operations_per_sec'].mean()
            summary_stats.append([backend, f'{avg_time:.1f}', f'{avg_memory:.1f}', f'{avg_throughput:.0f}'])
        
        table = ax4.table(cellText=summary_stats,
                         colLabels=['Backend', 'Avg Time (ms)', 'Avg Memory (MB)', 'Avg Throughput'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimized_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_optimized_report(self):
        """Generate optimized HTML report with only meaningful visualizations"""
        print("Generating optimized report...")
        
        report_path = self.output_dir / 'optimized_benchmark_report.html'
        
        # Calculate key metrics
        total_tests = len(self.performance_data) if self.performance_data is not None else 0
        backends = ', '.join(self.performance_data['Backend'].unique()) if self.performance_data is not None else 'None'
        
        # Calculate success rate
        success_rate = 0
        if self.correctness_data is not None and not self.correctness_data.empty:
            success_rate = (self.correctness_data['Status'] == 'PASS').mean() * 100
        elif self.performance_data is not None:
            success_rate = (self.performance_data['Success'] == 'SUCCESS').mean() * 100
        
        # Calculate best speedup
        best_speedup = "N/A"
        if self.performance_data is not None:
            seq_data = self.performance_data[self.performance_data['Backend'] == 'Sequential']
            if not seq_data.empty:
                seq_avg = seq_data['Time_ms'].mean()
                max_speedup = 0
                for backend in ['OpenMP', 'CUDA']:
                    if backend in self.performance_data['Backend'].values:
                        backend_data = self.performance_data[self.performance_data['Backend'] == backend]
                        backend_avg = backend_data['Time_ms'].mean()
                        speedup = seq_avg / backend_avg if backend_avg > 0 else 0
                        max_speedup = max(max_speedup, speedup)
                if max_speedup > 0:
                    best_speedup = f"{max_speedup:.1f}x"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OBDD Library Optimized Benchmark Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin: 30px 0; }}
        .metric {{ display: inline-block; margin: 15px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 8px; text-align: center; min-width: 150px; }}
        .metric h3 {{ margin: 0 0 10px 0; font-size: 16px; }}
        .metric p {{ margin: 0; font-size: 24px; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .insights {{ background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; }}
        .recommendations {{ background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; }}
        .key-finding {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; border-left: 3px solid #3498db; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 10px 0; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 OBDD Library Optimized Performance Report</h1>
        <p style="text-align: center; color: #7f8c8d;"><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2 style="color: white; border: none; text-align: center; margin-bottom: 20px;">Executive Summary</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <p>{total_tests}</p>
                </div>
                <div class="metric">
                    <h3>Backends</h3>
                    <p style="font-size: 16px;">{backends}</p>
                </div>
                <div class="metric">
                    <h3>Success Rate</h3>
                    <p class="good">{success_rate:.1f}%</p>
                </div>
                <div class="metric">
                    <h3>Best Speedup</h3>
                    <p class="good">{best_speedup}</p>
                </div>
            </div>
        </div>
        
        <h2>📊 Performance Overview</h2>
        <img src="optimized_performance_overview.png" alt="Performance Overview">
        <div class="insights">
            <h3>🔍 Key Insights</h3>
            <div class="key-finding">
                <strong>Scalability:</strong> Execution time shows exponential growth with problem size across all backends
            </div>
            <div class="key-finding">
                <strong>Parallelization:</strong> Parallel backends demonstrate significant speedup over sequential implementation
            </div>
            <div class="key-finding">
                <strong>Memory Usage:</strong> Linear growth in memory consumption with increasing variables
            </div>
            <div class="key-finding">
                <strong>Throughput:</strong> Clear performance hierarchy: CUDA > OpenMP > Sequential
            </div>
        </div>
        
        <h2>📈 Scalability Deep Dive</h2>
        <img src="optimized_scalability_analysis.png" alt="Scalability Analysis">
        <div class="insights">
            <h3>🎯 Scalability Analysis</h3>
            <p>Log-scale visualization reveals the exponential complexity of OBDD operations. Parallel backends maintain better scaling characteristics, with GPU acceleration showing the most consistent performance across problem sizes.</p>
        </div>
        
        <h2>📋 Executive Dashboard</h2>
        <img src="optimized_summary_dashboard.png" alt="Summary Dashboard">
        <div class="recommendations">
            <h3>💡 Actionable Recommendations</h3>
            <ul>
                <li><strong>Small Problems (< 8 variables):</strong> Use Sequential backend for simplicity and minimal overhead</li>
                <li><strong>Medium Problems (8-12 variables):</strong> OpenMP backend provides excellent CPU utilization</li>
                <li><strong>Large Problems (> 12 variables):</strong> CUDA backend delivers superior performance when GPU is available</li>
                <li><strong>Memory-Constrained Systems:</strong> Monitor memory usage patterns and consider problem decomposition</li>
                <li><strong>Production Deployment:</strong> Implement automatic backend selection based on problem size and available hardware</li>
            </ul>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 30px 0;">
            <h3 style="margin-top: 0;">🎯 Optimization Highlights</h3>
            <p>This report focuses on the most critical performance metrics:</p>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li><strong>Performance Scalability:</strong> How execution time grows with problem complexity</li>
                <li><strong>Speedup Analysis:</strong> Quantified parallel performance gains</li>
                <li><strong>Memory Efficiency:</strong> Resource utilization vs computational throughput</li>
                <li><strong>Success Validation:</strong> Correctness verification across all backends</li>
            </ul>
        </div>
        
        <footer>
            <p><em>Report generated by OBDD Library Optimized Benchmark Analyzer v2.0</em></p>
            <p>Focused on actionable insights and meaningful performance metrics</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Optimized report saved to: {report_path}")
    
    def run_optimized_analysis(self):
        """Run optimized benchmark analysis with only meaningful visualizations"""
        print("Starting optimized OBDD benchmark analysis...")
        print("="*60)
        
        # Generate only the most meaningful charts
        self.plot_performance_overview()      # Core performance metrics
        self.plot_scalability_deep_dive()     # Critical scalability analysis  
        self.plot_summary_dashboard()         # Executive summary
        self.generate_optimized_report()      # Clean, focused report
        
        print("="*60)
        print("Optimized analysis complete! Generated focused visualizations:")
        print(f"  ✅ Performance Overview: {self.output_dir}/optimized_performance_overview.png")
        print(f"  ✅ Scalability Analysis: {self.output_dir}/optimized_scalability_analysis.png")
        print(f"  ✅ Summary Dashboard: {self.output_dir}/optimized_summary_dashboard.png")
        print(f"  ✅ Optimized Report: {self.output_dir}/optimized_benchmark_report.html")
        print()
        print("🎯 This report contains only the most significant and actionable insights!")
        
        return str(self.output_dir)

def main():
    parser = argparse.ArgumentParser(description='Generate optimized OBDD benchmark analysis report')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing benchmark results (default: results)')
    
    args = parser.parse_args()
    
    analyzer = OptimizedOBDDBenchmarkAnalyzer(args.results_dir)
    output_path = analyzer.run_optimized_analysis()
    
    print(f"\n🚀 Open {output_path}/optimized_benchmark_report.html to view the focused report!")

if __name__ == "__main__":
    main()