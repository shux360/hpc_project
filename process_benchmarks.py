import csv
from collections import defaultdict
import statistics

# Read benchmark data
data = defaultdict(lambda: defaultdict(list))

with open('benchmark_results.csv', 'r', encoding='utf-16') as f:
    reader = csv.DictReader(f)
    for row in reader:
        impl = row['Implementation']
        threads = int(row['Threads'])
        time_ms = float(row['ExecutionTime_ms'])
        data[impl][threads].append(time_ms)

# Calculate averages and speedup
serial_baseline = statistics.mean(data['Serial'][1])
print(f"Serial baseline: {serial_baseline:.2f} ms\n")

results = []
for impl in ['Serial', 'OpenMP', 'Pthreads']:
    if impl in data:
        for threads in sorted(data[impl].keys()):
            avg_time = statistics.mean(data[impl][threads])
            speedup = serial_baseline / avg_time
            efficiency = (speedup / threads) * 100 if threads > 1 else 100
            
            results.append({
                'Implementation': impl,
                'Threads': threads,
                'Avg_Time': avg_time,
                'Speedup': speedup,
                'Efficiency': efficiency
            })
            
            print(f"{impl:12} | Threads: {threads:2d} | Time: {avg_time:7.2f} ms | Speedup: {speedup:5.2f}x | Efficiency: {efficiency:6.1f}%")

# Output for LaTeX table
print("\n=== LaTeX Table Data ===")
print("Implementation,1T,2T,4T,8T,16T")

for impl in ['Serial', 'OpenMP', 'Pthreads']:
    if impl in data:
        row = impl
        for threads in [1, 2, 4, 8, 16]:
            if threads in data[impl]:
                avg_time = statistics.mean(data[impl][threads])
                speedup = serial_baseline / avg_time if threads > 1 else 1.0
                row += f",{avg_time:.1f} ({speedup:.2f}x)"
            else:
                row += ",---"
        print(row)
