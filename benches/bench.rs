/*
 *
 * Copyright (c) 2025.
 * All rights reserved.
 *
 */

use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::vec::vec,
}
