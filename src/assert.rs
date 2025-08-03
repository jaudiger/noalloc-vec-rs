/*
 *
 * Copyright (c) 2025.
 * All rights reserved.
 *
 */

#[macro_export]
macro_rules! assert_lt {
    ($left:expr, $right:expr) => {
        assert!(
            $left < $right,
            "Left operand should be inferior to right operand"
        )
    };
}

#[macro_export]
macro_rules! assert_lte {
    ($left:expr, $right:expr) => {
        assert!(
            $left <= $right,
            "Left operand should be inferior or equal to right operand"
        )
    };
}

#[macro_export]
macro_rules! assert_gt {
    ($left:expr, $right:expr) => {
        assert!(
            $left > $right,
            "Left operand should be superior to right operand"
        )
    };
}

#[macro_export]
macro_rules! assert_gte {
    ($left:expr, $right:expr) => {
        assert!(
            $left >= $right,
            "Left operand should be superior or equal to right operand"
        )
    };
}
