import os
import time
import httpx

BASE = os.environ.get("BENCHMARK_BASE", "http://localhost:8000")


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
    elapsed = time.perf_counter() - t0
    return count / elapsed, count


def main():
    print("Part 1 — Basic inference benchmark (server on :8000, CPU)")
    print("Tip: run 'docker stats' in another terminal for CPU/RAM usage during benchmark.")
    print()
    print("Latency (ms), batch_size=1, 50 requests:")
    lat = run_latency(50, 1)
    s = sorted(lat)
    n = len(lat)
    print(f"  mean={sum(lat)/n:.2f} ms, p50={s[int(0.50*n)]:.2f} ms, p95={s[int(0.95*n)]:.2f} ms, p99={s[min(int(0.99*n), n-1)]:.2f} ms")
    print("Throughput (req/s), 10 sec:")
    tput, n = run_throughput(10, 1)
    print(f"  {tput:.2f} req/s ({n} requests)")


if __name__ == "__main__":
    main()
