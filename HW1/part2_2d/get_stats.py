import argparse

parser = argparse.ArgumentParser(description='converting stored values into stats')
parser.add_argument('--filename', type=str, help='Put filename to get stats')
args = parser.parse_args()

with open(args.filename, "r") as file:
    data = file.readlines()

costs = []
times = []

for line in data:
    parts = line.split()

    cost = float(parts[2])
    time = float(parts[4][:-3])

    costs.append(cost)
    times.append(time)

avg_cost = sum(costs) / len(costs) if costs else 0
avg_time = sum(times) / len(times) if times else 0

print(f"Average Cost: {avg_cost:.5f}")
print(f"Average Time: {avg_time:.2f} sec")
