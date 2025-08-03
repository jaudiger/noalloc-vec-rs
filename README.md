# No-allocation vector

## Instructions

This Rust library provides a no-allocation vector implementation. This vector is designed to work with a fixed maximum length, avoiding heap allocations (aka putting the vector on the stack) and making it suitable for embedded systems or other environments where dynamic memory allocation is not suitable.

## Usage

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

## CI / CD

The CI/CD pipeline is configured using GitHub Actions. The workflow is defined in the [`.github/workflows`](.github/workflows) folder:

- Static Analysis (source code, GitHub Actions)
- Tests (unit tests with code coverage generated)
- Code Audit (on each Cargo dependencies update, or run each day through CronJob)
- Deployment

Additionally, Dependabot is configured to automatically update dependencies (GitHub Actions, Cargo dependencies).

## Repository configuration

The settings of this repository are managed from the [gitops-deployments](https://github.com/jaudiger/gitops-deployments) repository using Terraform. The actual configuration applied is located in the Terraform module [`modules/github-repository`](https://github.com/jaudiger/gitops-deployments/tree/main/modules/github-repository).
