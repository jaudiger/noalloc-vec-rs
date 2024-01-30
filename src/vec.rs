/*
 *
 * Copyright (c) 2023-2024.
 * All rights reserved.
 *
 */

use core::mem::size_of;
use core::mem::ManuallyDrop;
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

pub struct IntoIter<T, const MAX_LENGTH: usize> {
    vec: Vec<T, MAX_LENGTH>,
    next: usize,
}

impl<T, const MAX_LENGTH: usize> IntoIter<T, MAX_LENGTH> {
    #[must_use]
    pub fn new(vec: Vec<T, MAX_LENGTH>) -> Self {
        Self { vec, next: 0 }
    }
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

    #[allow(clippy::result_unit_err)]
    pub fn push(&mut self, value: T) -> Result<(), ()> {
        if self.length < MAX_LENGTH {
            self.push_unchecked(value);
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn push_unchecked(&mut self, value: T) {
        self.array[self.length].write(value);
        self.length += 1;
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for elem in iter {
            self.push_unchecked(elem);
        }
    }

    #[allow(clippy::result_unit_err)]
    pub fn write(&mut self, index: usize, value: T) -> Result<(), ()> {
        // Make sure all the previous bytes are initialized before reading the array
        if index < MAX_LENGTH {
            self.array[index].write(value);
            if index >= self.length {
                self.length = index + 1;
            }
            Ok(())
        } else {
            Err(())
        }
    }

    #[allow(clippy::result_unit_err)]
    pub fn write_slice(&mut self, index: usize, value: &[T]) -> Result<(), ()>
    where
        T: Copy,
    {
        // Make sure all the previous bytes are initialized before reading the array
        if index + value.len() <= MAX_LENGTH {
            let mut buffer_index = index;
            for byte in value {
                self.array[buffer_index].write(*byte);
                buffer_index += 1;
            }
            if buffer_index >= self.length {
                self.length = buffer_index;
            }
            Ok(())
        } else {
            Err(())
        }
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
    pub const fn remaining_len(&self) -> usize {
        MAX_LENGTH - self.length
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
            Some(self.get_unchecked(index))
        } else {
            None
        }
    }

    #[must_use]
    pub fn get_unchecked(&self, index: usize) -> T {
        unsafe { self.array.get_unchecked(index).as_ptr().read() }
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

    #[must_use]
    fn from_array_unchecked<const LENGTH: usize>(from_array: [T; LENGTH]) -> Self {
        let mut vec = Self::new();

        // Do not drop the elements of the array, since we're moving them into the vector
        for byte in ManuallyDrop::new(from_array).iter() {
            vec.push_unchecked(unsafe { ptr::read(byte) });
        }

        vec
    }

    #[must_use]
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

    #[must_use]
    fn to_uint(&self) -> u64
    where
        T: Into<u8>,
    {
        let mut value = 0;
        let mut index = 0;
        while index < self.len() {
            let byte = self.get_unchecked(index).into();

            value |= u64::from(byte) << (index * 8);

            index += 1;
        }

        value
    }
}

impl<T, const MAX_LENGTH: usize> Default for Vec<T, MAX_LENGTH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const LENGTH: usize> Drop for Vec<T, LENGTH> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
        }
    }
}

impl<'a, T, const MAX_LENGTH: usize> IntoIterator for &'a Vec<T, MAX_LENGTH> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const MAX_LENGTH: usize> Iterator for IntoIter<T, MAX_LENGTH> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.vec.len() {
            let value = self.vec.get_unchecked(self.next);
            self.next += 1;

            Some(value)
        } else {
            None
        }
    }
}

impl<T, const MAX_LENGTH: usize> IntoIterator for Vec<T, MAX_LENGTH> {
    type Item = T;
    type IntoIter = IntoIter<T, MAX_LENGTH>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<TA, TB, const MAX_LENGTH_A: usize, const MAX_LENGTH_B: usize> PartialEq<Vec<TB, MAX_LENGTH_B>>
    for Vec<TA, MAX_LENGTH_A>
where
    TA: PartialEq<TB>,
{
    fn eq(&self, other: &Vec<TB, MAX_LENGTH_B>) -> bool {
        <[TA]>::eq(self, &**other)
    }
}

impl<T, const MAX_LENGTH: usize> Eq for Vec<T, MAX_LENGTH> where T: Eq {}

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

impl<T, const LENGTH: usize, const MAX_LENGTH: usize> From<[T; LENGTH]> for Vec<T, MAX_LENGTH> {
    fn from(values: [T; LENGTH]) -> Self {
        // Build time assertion
        Assert::<LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_array_unchecked(values)
    }
}

impl<T: Copy, const LENGTH: usize, const MAX_LENGTH: usize> From<&[T; LENGTH]>
    for Vec<T, MAX_LENGTH>
{
    fn from(values: &[T; LENGTH]) -> Self {
        // Build time assertion
        Assert::<LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_slice_unchecked(values)
    }
}

impl<const MAX_LENGTH: usize> From<u8> for Vec<u8, MAX_LENGTH> {
    fn from(value: u8) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u8>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

impl<const MAX_LENGTH: usize> From<u16> for Vec<u8, MAX_LENGTH> {
    fn from(value: u16) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u16>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

impl<const MAX_LENGTH: usize> From<u32> for Vec<u8, MAX_LENGTH> {
    fn from(value: u32) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u32>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

impl<const MAX_LENGTH: usize> From<u64> for Vec<u8, MAX_LENGTH> {
    fn from(value: u64) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u64>();
        Assert::<VALUE_LENGTH, MAX_LENGTH>::less_than_or_equal();

        Self::from_uint_unchecked(value, VALUE_LENGTH)
    }
}

impl<T, const MAX_LENGTH: usize> Deref for Vec<T, MAX_LENGTH> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u8
where
    T: Into<Self>,
{
    #[allow(clippy::cast_possible_truncation)]
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u16
where
    T: Into<u8>,
{
    #[allow(clippy::cast_possible_truncation)]
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u32
where
    T: Into<u8>,
{
    #[allow(clippy::cast_possible_truncation)]
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u64
where
    T: Into<u8>,
{
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

impl<'a, T, const N: usize> Extend<&'a T> for Vec<T, N>
where
    T: 'a + Copy,
{
    // Check left capacity before using this method
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        self.extend(iter.into_iter().copied());
    }
}

impl<T, const N: usize> Extend<T> for Vec<T, N>
where
    T: Copy,
{
    // Check left capacity before using this method
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.extend(iter);
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
    fn test_vec_push_slice() {
        let mut vec = Vec::<u8, 3>::new();
        let array: [u8; 3] = [1, 2, 3];

        vec.extend(array.iter().copied());
        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_write() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Ok(()), vec.write(0, 1));
        assert_eq!(1, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_write_out_of_bound() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Err(()), vec.write(3, 1));
    }

    #[test]
    fn test_vec_write_slice() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Ok(()), vec.write_slice(0, &[1, 2, 3]));
        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_write_slice_out_of_bound() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Err(()), vec.write_slice(1, &[1, 2, 3]));
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
    fn test_vec_try_from_array_as_slice() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_slice().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_array_as_slice_shorter_than_vec_size() {
        let vec: Vec<u8, 8> = [1, 2, 3].as_slice().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_small_vec_try_from_array_as_slice_should_failed() {
        let vec_result: Result<Vec<u8, 1>, _> = [1, 2, 3].as_slice().try_into();

        assert!(vec_result.is_err());
    }

    #[test]
    fn test_vec_from_array() {
        let vec: Vec<u8, 3> = Vec::from(&[1, 2, 3]);

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_from_array_same_size_as_vec() {
        let vec: Vec<u8, 3> = [1, 2, 3].into();

        assert_eq!(3, vec.len());
        assert_eq!(Some(1), vec.get(0));
        assert_eq!(Some(2), vec.get(1));
        assert_eq!(Some(3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_from_u8() {
        let vec: Vec<u8, 8> = 0xffu8.into();

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
    fn test_vec_from_u16() {
        let vec: Vec<u8, 8> = 0xff00u16.into();

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
    fn test_vec_from_number_shorter_than_real_u16() {
        let vec: Vec<u8, 8> = 0x00ffu16.into();

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
    fn test_vec_from_u32() {
        let vec: Vec<u8, 8> = 0xff00_ff00_u32.into();

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
    fn test_vec_from_u64() {
        let vec: Vec<u8, 8> = 0xff00_ff00_ff00_ff00_u64.into();

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
    fn test_u8_from_vec() {
        let vec: Vec<u8, 1> = Vec::from([0x2A]);
        let value = u8::from(&vec);

        assert_eq!(42, value);
    }

    #[test]
    fn test_u16_from_vec() {
        let vec: Vec<u8, 2> = Vec::from([0xD2, 0x04]);
        let value = u16::from(&vec);

        assert_eq!(1234, value);
    }

    #[test]
    fn test_u32_from_vec() {
        let vec: Vec<u8, 4> = Vec::from([0x52, 0xAA, 0x08, 0x00]);
        let value = u32::from(&vec);

        assert_eq!(567_890, value);
    }

    #[test]
    fn test_u64_from_vec() {
        let vec: Vec<u8, 8> = Vec::from([0x08, 0x1A, 0x99, 0xBE, 0x1C, 0x00, 0x00, 0x00]);
        let value = u64::from(&vec);

        assert_eq!(123_456_789_000, value);
    }

    #[test]
    fn test_deref_with_empty_vec() {
        let vec = Vec::<u8, 1>::new();

        let array = &*vec;

        assert_eq!(0, array.len());
    }

    #[test]
    fn test_into_iter_vec_with_for_loop() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_slice().try_into().unwrap();

        // Using for loop
        vec.into_iter().for_each(|value| {
            assert!(matches!(value, 1..=3));
        });
    }

    #[test]
    fn test_into_iter_vec_with_iterator() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_slice().try_into().unwrap();

        // Using iterator
        let mut into_iter = vec.into_iter();
        assert_eq!(Some(1), into_iter.next());
        assert_eq!(Some(2), into_iter.next());
        assert_eq!(Some(3), into_iter.next());
    }
}
