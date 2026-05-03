---
layout: post
title: "Designing a Rate Limiter"
date: 2026-05-01
category: System Design
description: A walkthrough of common rate limiting algorithms — token bucket, leaky bucket, and sliding window — with implementation notes.
---

## Why Rate Limiting?

Rate limiting protects services from being overwhelmed by controlling the rate at which clients can make requests. It is essential for:

- Preventing abuse and DoS attacks
- Enforcing fair use across tenants
- Protecting downstream dependencies

## Algorithms

### Token Bucket

Each client has a bucket of tokens with capacity $C$. Tokens are added at a fixed rate $r$ (tokens/second). Each request consumes one token. If the bucket is empty, the request is rejected.

**Properties:**
- Allows bursting up to $C$ requests
- Smooth average throughput of $r$ requests/second

```python
import time

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

### Leaky Bucket

Requests enter a queue (the bucket) and are processed at a fixed rate $r$. Requests that overflow the bucket are dropped.

Unlike token bucket, leaky bucket enforces a strictly constant output rate — no bursting.

### Sliding Window Counter

Divide time into fixed windows (e.g., 1-minute intervals). Track request counts per window. The allowed count for the current moment is:

$$
\text{count} = \text{count}_{\text{prev}} \times \frac{W - t_{\text{elapsed}}}{W} + \text{count}_{\text{curr}}
$$

where $W$ is the window size and $t_{\text{elapsed}}$ is time elapsed in the current window.

This approximates a true sliding window with $O(1)$ space.

## Distributed Rate Limiting

Single-node implementations break in distributed environments. Common approaches:

1. **Centralized store (Redis):** Atomic `INCR` + `EXPIRE` on a key per client. Lua scripts for atomic token bucket updates.
2. **Sticky routing:** Route each client to the same backend node — eliminates coordination.
3. **Approximate counting:** Accept small inaccuracies in exchange for no coordination overhead.

## Where to Place It

| Placement | Pros | Cons |
|-----------|------|------|
| API Gateway | Centralized, easy to manage | Single point of failure |
| Service mesh (sidecar) | Per-service granularity | More complex deployment |
| Application layer | Full business logic access | Code duplication |

## Summary

For most services, a **token bucket in Redis** with Lua scripting is the standard choice. It allows bursting, is easy to reason about, and scales well with a shared Redis cluster.
