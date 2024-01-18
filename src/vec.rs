/*
 *
 * Copyright (c) 2023-2024.
 * All rights reserved.
 *
 */

use core::convert::Infallible;
use core::mem::size_of;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ptr;
use core::slice;
use core::usize;

use crate::assert::Assert;

#[derive(Debug)]
pub struct Vec<T, const MAX_LENGTH: usize> {
    array: [MaybeUninit<T>; MAX_LENGTH],
    length: usize,
}

impl<T, const MAX_LENGTH: usize> Vec<T, MAX_LENGTH> {
    const ARRAY_INIT_VALUE: MaybeUninit<T> = MaybeUninit::uninit();

    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            array: [Self::ARRAY_INIT_VALUE; MAX_LENGTH],
            length: 0,
        }
    }

    pub fn push(&mut self, value: T) -> Result<(), ()> {
        if self.length < MAX_LENGTH {
            self.push_unchecked(value);
            Ok(())
        } else {
            Err(())
        }
    }

    fn push_unchecked(&mut self, value: T) {
        self.array[self.length].write(value);
        self.length += 1;
    }

    #[must_use]
    pub fn pop(&mut self) -> Option<T> {
        if self.length > 0 {
            self.length -= 1;
            Some(unsafe { self.array.get_unchecked(self.length).as_ptr().read() })
        } else {
            None
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.length = 0;
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.length {
            Some(unsafe { self.array.get_unchecked(index).as_ptr().read() })
        } else {
            None
        }
    }

    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<T> {
        if index < self.length {
            Some(unsafe { self.array.get_unchecked_mut(index).as_mut_ptr().read() })
        } else {
            None
        }
    }

    #[must_use]
    const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.array.as_ptr().cast::<T>(), self.length) }
    }

    #[must_use]
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.array.as_mut_ptr().cast::<T>(), self.length) }
    }

    fn from_array_unchecked<const LENGTH: usize>(from_array: [T; LENGTH]) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::new();

        for byte in &from_array {
            vec.push_unchecked(*byte);
        }

        vec
    }

    fn from_slice_unchecked(from_slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::new();

        for byte in from_slice {
            vec.push_unchecked(*byte);
        }

        vec
    }

    #[must_use]
    fn from_uint_unchecked(from_value: u64, max_length: usize) -> Self
    where
        T: From<u8>,
    {
        let mut vec = Self::new();

        let mut value = from_value;
        let mut index = 0;
        while index < max_length {
            let byte = (value & 0xff) as u8;

            vec.push_unchecked(byte.into());

            // Shift the value to the right
            value >>= 8;

            index += 1;
        }

        vec
    }
}

impl<T, const MAX_LENGTH: usize> Default for Vec<T, MAX_LENGTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for Vec<T, N> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vec<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Copy, const LENGTH: usize, const MAX_LENGTH: usize> TryFrom<[T; LENGTH]>
    for Vec<T, MAX_LENGTH>
{
    type Error = Infallible;

    fn try_from(values: [T; LENGTH]) -> Result<Self, Self::Error> {
        // Build time check
        Assert::<LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_array_unchecked(values))
    }
}

impl<T: Copy, const MAX_LENGTH: usize> TryFrom<&[T]> for Vec<T, MAX_LENGTH> {
    type Error = ();

    fn try_from(values: &[T]) -> Result<Self, Self::Error> {
        // Runtime check
        if values.len() > MAX_LENGTH {
            return Err(());
        }

        Ok(Self::from_slice_unchecked(values))
    }
}

impl<T: Copy, const LENGTH: usize, const MAX_LENGTH: usize> TryFrom<&[T; LENGTH]>
    for Vec<T, MAX_LENGTH>
{
    type Error = Infallible;

    fn try_from(values: &[T; LENGTH]) -> Result<Self, Self::Error> {
        // Build time check
        Assert::<LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_slice_unchecked(values))
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u8> for Vec<u8, MAX_LENGTH> {
    type Error = Infallible;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        // Build time check
        const VALUE_LENGTH: usize = size_of::<u8>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH))
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u16> for Vec<u8, MAX_LENGTH> {
    type Error = Infallible;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        // Build time check
        const VALUE_LENGTH: usize = size_of::<u16>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH))
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u32> for Vec<u8, MAX_LENGTH> {
    type Error = Infallible;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        // Build time check
        const VALUE_LENGTH: usize = size_of::<u32>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH))
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u64> for Vec<u8, MAX_LENGTH> {
    type Error = Infallible;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        // Build time check
        const VALUE_LENGTH: usize = size_of::<u64>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Ok(Self::from_uint_unchecked(value, VALUE_LENGTH))
    }
}

impl<T, const MAX_LENGTH: usize> Deref for Vec<T, MAX_LENGTH> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::Vec;

    #[test]
    fn test_vec_new() {
        let vec = Vec::<u8, 1>::new();

        assert_eq!(0, vec.len());
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vec_push() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(1, vec.len());
        assert!(!vec.is_empty());
    }

    #[test]
    fn test_vec_push_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(Err(()), vec.push(2));
    }

    #[test]
    fn test_vec_pop() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(Some(1), vec.pop());
        assert_eq!(0, vec.len());
        assert_eq!(None, vec.pop());
    }

    #[test]
    fn test_vec_clear() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert!(!vec.is_empty());

        vec.clear();

        assert_eq!(0, vec.len());
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vec_get() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(1, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_get_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_get_mut() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(1, vec.len());
        assert_eq!(Some(1), vec.get_mut(0));
        assert_eq!(None, vec.get_mut(1));
    }

    #[test]
    fn test_vec_get_mut_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(None, vec.get_mut(1));
    }

    #[test]
    fn test_vec_try_from_array_as_ref() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_ref().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_array_as_ref_shorter_than_vec_size() {
        let vec: Vec<u8, 8> = [1, 2, 3].as_ref().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_small_vec_try_from_array_as_ref_should_failed() {
        let vec_result: Result<Vec<u8, 1>, _> = [1, 2, 3].as_ref().try_into();

        assert!(vec_result.is_err());
    }

    #[test]
    fn test_vec_try_from_array() {
        let vec: Vec<u8, 3> = Vec::try_from(&[1, 2, 3]).unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_array_same_size_as_vec() {
        let vec: Vec<u8, 3> = [1, 2, 3].try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_u8() {
        let vec: Vec<u8, 8> = 0xffu8.try_into().unwrap();

        assert_eq!(1, vec.len());
        assert_eq!(Some(0xff), vec.get(0));
        assert_eq!(None, vec.get(1));
        assert_eq!(None, vec.get(2));
        assert_eq!(None, vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_u16() {
        let vec: Vec<u8, 8> = 0xff00u16.try_into().unwrap();

        assert_eq!(2, vec.len());
        assert_eq!(Some(0x00), vec.get(0));
        assert_eq!(Some(0xff), vec.get(1));
        assert_eq!(None, vec.get(2));
        assert_eq!(None, vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_number_shorter_than_real_u16() {
        let vec: Vec<u8, 8> = 0x00ffu16.try_into().unwrap();

        assert_eq!(2, vec.len());
        assert_eq!(Some(0xff), vec.get(0));
        assert_eq!(Some(0x00), vec.get(1));
        assert_eq!(None, vec.get(2));
        assert_eq!(None, vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_u32() {
        let vec: Vec<u8, 8> = 0xff00_ff00_u32.try_into().unwrap();

        assert_eq!(4, vec.len());
        assert_eq!(Some(0x00), vec.get(0));
        assert_eq!(Some(0xff), vec.get(1));
        assert_eq!(Some(0x00), vec.get(2));
        assert_eq!(Some(0xff), vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_u64() {
        let vec: Vec<u8, 8> = 0xff00_ff00_ff00_ff00_u64.try_into().unwrap();

        assert_eq!(8, vec.len());
        assert_eq!(Some(0x00), vec.get(0));
        assert_eq!(Some(0xff), vec.get(1));
        assert_eq!(Some(0x00), vec.get(2));
        assert_eq!(Some(0xff), vec.get(3));
        assert_eq!(Some(0x00), vec.get(4));
        assert_eq!(Some(0xff), vec.get(5));
        assert_eq!(Some(0x00), vec.get(6));
        assert_eq!(Some(0xff), vec.get(7));
    }

    #[test]
    fn test_deref_with_empty_vec() {
        let vec = Vec::<u8, 1>::new();

        let array = &*vec;

        assert_eq!(0, array.len());
    }

    #[test]
    fn test_into_iter_vec() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_ref().try_into().unwrap();

        // Using for loop
        vec.into_iter().for_each(|value| {
            assert!(matches!(value, 1..=3));
        });

        // Using iterator
        let mut into_iter = vec.into_iter();
        assert_eq!(Some(&1), into_iter.next());
        assert_eq!(Some(&2), into_iter.next());
        assert_eq!(Some(&3), into_iter.next());
    }
}
