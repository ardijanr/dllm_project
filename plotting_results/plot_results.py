import matplotlib.pyplot as plt

def load_csv(file_path):
    data = {}
    with open(file_path, 'r') as file:
        headers = file.readline().strip().split(',')
        for header in headers:
            data[header.strip()] = []

        for line in file:
            values = line.strip().split(',')
            for i, value in enumerate(values):
                data[headers[i].strip()].append(float(value.strip()))
    return data

def create_box_plot(data):
    values = [v for v in data.values()]
    labels = list(data.keys())

    averages = [sum(v) / len(v) for v in values]
    variances = [sum((x - avg) ** 2 for x in v) / len(v) for v, avg in zip(values, averages)]

    plt.figure(figsize=(10, 10))
    box = plt.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
    plt.title('Benchmark results')
    plt.xlabel('')
    plt.ylabel('Time in seconds')
    plt.grid(True)

    for i, (avg, var) in enumerate(zip(averages, variances), start=1):
        plt.text(i, max(values[i-1]), f'Avg: {avg:.2f}\nVar: {var:.2f}',
                 horizontalalignment='center', fontsize=12, fontweight='bold', color='blue')

    plt.savefig('./benchmark_results.png')

file_path = "results.csv"
data = load_csv(file_path)
create_box_plot(data)
