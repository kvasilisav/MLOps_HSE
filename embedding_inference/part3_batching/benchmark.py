import asyncio
import os
import time
import httpx

BASE = os.environ.get("BENCHMARK_BASE", "http://localhost:8002")


async def run_concurrent(n_clients: int, requests_per_client: int):
    async with httpx.AsyncClient(timeout=60.0) as c:
        async def one_client():
            for _ in range(requests_per_client):
                r = await c.post(
                    BASE + "/embed",
                    json={"texts": ["Текст запроса."], "prompt": "categorize: "},
                )
                r.raise_for_status()

        t0 = time.perf_counter()
        await asyncio.gather(*[one_client() for _ in range(n_clients)])
        elapsed = time.perf_counter() - t0
    total = n_clients * requests_per_client
    return total / elapsed, total, elapsed


def run_latency(n_requests: int = 50):
    latencies = []
    with httpx.Client(timeout=60.0) as c:
        for _ in range(n_requests):
            t0 = time.perf_counter()
            r = c.post(
                BASE + "/embed",
                json={"texts": ["Тестовый текст."], "prompt": "categorize: "},
            )
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def main():
    print("Part 3 — Dynamic batching benchmark (port 8002)")
    print("Latency (ms), 50 sequential requests:")
    lat = run_latency(50)
    print(f"  mean={sum(lat)/len(lat):.2f} ms, p95={sorted(lat)[int(0.95*len(lat))]:.2f} ms")
    print("Throughput: 10 concurrent clients, 20 req each (200 total):")
    tput, n, el = asyncio.run(run_concurrent(10, 20))
    print(f"  {tput:.2f} req/s ({n} requests in {el:.2f}s)")


if __name__ == "__main__":
    main()
