import os
import time
import httpx

BASE = os.environ.get("BENCHMARK_BASE", "http://localhost:8001")


def run_latency(n_requests: int = 50, batch_size: int = 1):
    texts = ["Тестовый текст для эмбеддинга."] * batch_size
    latencies = []
    with httpx.Client(timeout=60.0) as c:
        for _ in range(n_requests):
            t0 = time.perf_counter()
            r = c.post(f"{BASE}/embed", json={"texts": texts, "prompt": "categorize: "})
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def run_throughput(duration_sec: float = 10, batch_size: int = 1):
    texts = ["Текст для бенчмарка."] * batch_size
    count = 0
    t0 = time.perf_counter()
    with httpx.Client(timeout=60.0) as c:
        while time.perf_counter() - t0 < duration_sec:
            r = c.post(f"{BASE}/embed", json={"texts": texts, "prompt": "categorize: "})
            r.raise_for_status()
            count += 1
    return count / (time.perf_counter() - t0), count


def main():
    print("Part 2 — ONNX inference benchmark (port 8001)")
    print("Latency (ms), batch_size=1, 50 requests:")
    lat = run_latency(50, 1)
    print(f"  mean={sum(lat)/len(lat):.2f} ms, p95={sorted(lat)[int(0.95*len(lat))]:.2f} ms")
    print("Throughput (req/s), 10 sec:")
    tput, n = run_throughput(10, 1)
    print(f"  {tput:.2f} req/s ({n} requests)")


if __name__ == "__main__":
    main()
