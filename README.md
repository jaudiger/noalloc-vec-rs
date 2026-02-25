# No-allocation vector

## Getting Started

This Rust library provides a no-allocation vector implementation. This vector is designed to work with a fixed maximum length, avoiding heap allocations (aka putting the vector on the stack) and making it suitable for embedded systems or other environments where dynamic memory allocation is not suitable.

### Usage

```rust
use noalloc_vec_rs::Vec;

const MAX_LENGTH: usize = 10;
let mut vec = Vec::<u32, MAX_LENGTH>::new();

vec.push(42).unwrap();
vec.push(43).unwrap();

assert_eq!(vec.len(), 2);
assert_eq!(vec.get(0), Some(&42));
assert_eq!(vec.get(1), Some(&43));
```
