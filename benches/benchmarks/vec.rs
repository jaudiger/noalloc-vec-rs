/*
 *
 * Copyright (c) 2024.
 * All rights reserved.
 *
 */

use std::hint::black_box;

use criterion::Criterion;
use criterion::criterion_group;
use utils::vec::Vec;

fn vec_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec");

    let buffer: [u8; 8] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

    // From benchmarks
    group.bench_function("From buffer as slice", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::try_from(black_box(buffer.as_slice())).unwrap()));
    });
    group.bench_function("From &buffer", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(&buffer))));
    });
    group.bench_function("From buffer", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(buffer))));
    });
    group.bench_function("From u8", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(0xffu8))));
    });
    group.bench_function("From u16", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(0xff00u16))));
    });
    group.bench_function("From u32", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(0xff00_ff00_u32))));
    });
    group.bench_function("From u64", |b| {
        b.iter(|| black_box(Vec::<u8, 8>::from(black_box(0xff00_ff00_ff00_ff00_u64))));
    });
}

criterion_group!(vec, vec_benchmarks,);
