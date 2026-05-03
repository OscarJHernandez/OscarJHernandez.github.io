---
layout: post
title: "The CAP Theorem: Consistency, Availability, and Partition Tolerance"
date: 2026-04-15
category: System Design
description: A rigorous look at the CAP theorem and what it actually means for distributed system design decisions.
---

## Overview

The CAP theorem, proved by Eric Brewer in 2000, states that a distributed data store can guarantee at most **two** of the following three properties simultaneously:

- **Consistency (C):** Every read receives the most recent write or an error.
- **Availability (A):** Every request receives a (non-error) response, without guaranteeing it is the most recent.
- **Partition Tolerance (P):** The system continues to operate despite an arbitrary number of messages being dropped or delayed between nodes.

In practice, network partitions are not optional — they happen. Therefore, the real trade-off for distributed systems is between **C** and **A** in the presence of a partition.

## Formal Statement

Let a distributed system consist of nodes $N = \{n_1, n_2, \ldots, n_k\}$. A *partition* is a network failure that splits $N$ into disjoint subsets such that messages between subsets are lost.

Given a partition event $P$:

- A **CP system** chooses to return an error (or block) rather than return stale data.
- An **AP system** chooses to return possibly stale data rather than return an error.

## Why You Cannot Have All Three

Suppose nodes $n_1$ and $n_2$ are partitioned. A client writes value $v_1$ to $n_1$. Now another client reads from $n_2$.

$$
\text{If Availability} \Rightarrow n_2 \text{ must respond.}
$$

$$
\text{If Consistency} \Rightarrow n_2 \text{ must return } v_1.
$$

But $n_2$ cannot know about $v_1$ because the partition blocks the replication. Contradiction. $\square$

## Practical Implications

| System | Trade-off | Example |
|--------|-----------|---------|
| HBase | CP | Returns error under partition |
| Cassandra | AP | Returns stale data under partition |
| Zookeeper | CP | Leader election, blocks on partition |
| DynamoDB | AP (tunable) | Eventual consistency by default |

## PACELC: The Full Picture

CAP only describes behavior *during* a partition. The **PACELC** model extends this:

- **PAC** = During a **P**artition, trade **A**vailability vs **C**onsistency.
- **ELC** = **E**lse (no partition), trade **L**atency vs **C**onsistency.

Most production systems must be reasoned about across both dimensions.

## Conclusion

The CAP theorem is a theorem about trade-offs, not a recipe. Understanding which consistency guarantees your system actually needs — and which it can safely relax — is more valuable than picking a quadrant.
