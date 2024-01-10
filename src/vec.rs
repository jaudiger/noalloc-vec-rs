/*
 *
 * Copyright (c) 2023.
 * All rights reserved.
 *
 */

use core::mem::MaybeUninit;
use core::usize;

pub struct Vec<T, const MAX_LENGTH: usize> {
    array: [T; MAX_LENGTH],
    length: usize,
}

impl<T, const MAX_LENGTH: usize> Vec<T, MAX_LENGTH> {
    #[inline]
    pub fn new() -> Self {
        Self {
            array: unsafe { MaybeUninit::uninit().assume_init_read() },
            length: 0,
        }
    }

    pub fn push(&mut self, value: T) -> Result<(), ()> {
        if self.length < MAX_LENGTH {
            self.array[self.length] = value;
            self.length += 1;
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn pop(&mut self) -> Option<T>
    where
        T: Copy,
    {
        if self.length > 0 {
            self.length -= 1;
            Some(self.array[self.length])
        } else {
            None
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.length = 0;
    }

    pub const fn get(&self, index: usize) -> Option<&T> {
        if index < self.length {
            Some(&self.array[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.length {
            Some(&mut self.array[index])
        } else {
            None
        }
    }

    fn from_uint(array: &mut [MaybeUninit<u8>], from_value: u64) -> Result<usize, ()> {
        let mut value = from_value;
        let mut index = 0;
        let mut real_index = 0;
        while index < 8 {
            let byte = (value & 0xff) as u8;

            // The byte cannot be saved to the array
            if byte != 0 {
                if index >= MAX_LENGTH {
                    return Err(());
                }

                real_index = index + 1;
            }

            array[index].write(byte);

            // Shift the value to the right
            value = value >> 8;

            index += 1;
        }

        Ok(real_index)
    }
}

impl<T: Copy, const MAX_LENGTH: usize> TryFrom<&[T]> for Vec<T, MAX_LENGTH> {
    type Error = ();

    fn try_from(values: &[T]) -> Result<Self, Self::Error> {
        if values.len() > MAX_LENGTH {
            return Err(());
        }

        let mut array: [MaybeUninit<T>; MAX_LENGTH] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Do not initialize the array

        values.iter().enumerate().for_each(|(i, byte)| {
            array[i].write(*byte);
        });

        Ok(Self {
            array: unsafe {
                (*(&MaybeUninit::new(array) as *const _ as *const MaybeUninit<_>))
                    .assume_init_read()
            },
            length: values.len(),
        })
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u8> for Vec<u8, MAX_LENGTH> {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        let mut array: [MaybeUninit<u8>; MAX_LENGTH] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Do not initialize the array

        let length = Vec::<u8, MAX_LENGTH>::from_uint(&mut array, value as u64)?;

        Ok(Self {
            array: unsafe {
                (*(&MaybeUninit::new(array) as *const _ as *const MaybeUninit<_>))
                    .assume_init_read()
            },
            length,
        })
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u16> for Vec<u8, MAX_LENGTH> {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        let mut array: [MaybeUninit<u8>; MAX_LENGTH] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Do not initialize the array

        let length = Vec::<u8, MAX_LENGTH>::from_uint(&mut array, value as u64)?;

        Ok(Self {
            array: unsafe {
                (*(&MaybeUninit::new(array) as *const _ as *const MaybeUninit<_>))
                    .assume_init_read()
            },
            length,
        })
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u32> for Vec<u8, MAX_LENGTH> {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let mut array: [MaybeUninit<u8>; MAX_LENGTH] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Do not initialize the array

        let length = Vec::<u8, MAX_LENGTH>::from_uint(&mut array, value as u64)?;

        Ok(Self {
            array: unsafe {
                (*(&MaybeUninit::new(array) as *const _ as *const MaybeUninit<_>))
                    .assume_init_read()
            },
            length,
        })
    }
}

impl<const MAX_LENGTH: usize> TryFrom<u64> for Vec<u8, MAX_LENGTH> {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let mut array: [MaybeUninit<u8>; MAX_LENGTH] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Do not initialize the array

        let length = Vec::<u8, MAX_LENGTH>::from_uint(&mut array, value as u64)?;

        Ok(Self {
            array: unsafe {
                (*(&MaybeUninit::new(array) as *const _ as *const MaybeUninit<_>))
                    .assume_init_read()
            },
            length,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::Vec;

    #[test]
    fn test_vec_new() {
        let vec: Vec<u8, 1> = Vec::new();

        assert_eq!(0, vec.len());
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vec_push() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(1, vec.len());
        assert!(!vec.is_empty());
    }

    #[test]
    fn test_vec_push_out_of_bound() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(Err(()), vec.push(2));
    }

    #[test]
    fn test_vec_pop() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));
        assert_eq!(Some(1), vec.pop());
        assert_eq!(0, vec.len());
        assert_eq!(None, vec.pop());
    }

    #[test]
    fn test_vec_clear() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));
        assert!(!vec.is_empty());

        vec.clear();

        assert_eq!(0, vec.len());
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vec_get() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(1, vec.len());
        assert_eq!(Some(&1), vec.get(0));
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_get_out_of_bound() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_get_mut() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(1, vec.len());
        assert_eq!(Some(&mut 1), vec.get_mut(0));
        assert_eq!(None, vec.get_mut(1));
    }

    #[test]
    fn test_vec_get_mut_out_of_bound() {
        let mut vec: Vec<u8, 1> = Vec::new();

        assert_eq!(Ok(()), vec.push(1));

        assert_eq!(None, vec.get_mut(1));
    }

    #[test]
    fn test_vec_try_from_buffer() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_ref().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.get(0));
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_buffer_shorter_than_vec_size() {
        let vec: Vec<u8, 8> = [1, 2, 3].as_ref().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.get(0));
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_small_vec_try_from_buffer_should_failed() {
        let vec_result: Result<Vec<u8, 1>, _> = [1, 2, 3].as_ref().try_into();

        assert!(vec_result.is_err());
    }

    #[test]
    fn test_vec_try_from_u8() {
        let vec: Vec<u8, 8> = 0xffu8.try_into().unwrap();

        assert_eq!(1, vec.len());
        assert_eq!(Some(&0xff), vec.get(0));
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
        assert_eq!(Some(&0x00), vec.get(0));
        assert_eq!(Some(&0xff), vec.get(1));
        assert_eq!(None, vec.get(2));
        assert_eq!(None, vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_number_shorter_than_u16() {
        let vec: Vec<u8, 8> = 0x00ffu16.try_into().unwrap();

        assert_eq!(1, vec.len());
        assert_eq!(Some(&0xff), vec.get(0));
        assert_eq!(None, vec.get(1));
        assert_eq!(None, vec.get(2));
        assert_eq!(None, vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_u32() {
        let vec: Vec<u8, 8> = 0xff00ff00u32.try_into().unwrap();

        assert_eq!(4, vec.len());
        assert_eq!(Some(&0x00), vec.get(0));
        assert_eq!(Some(&0xff), vec.get(1));
        assert_eq!(Some(&0x00), vec.get(2));
        assert_eq!(Some(&0xff), vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_try_from_u64() {
        let vec: Vec<u8, 8> = 0xff00ff00ff00ff00u64.try_into().unwrap();

        assert_eq!(8, vec.len());
        assert_eq!(Some(&0x00), vec.get(0));
        assert_eq!(Some(&0xff), vec.get(1));
        assert_eq!(Some(&0x00), vec.get(2));
        assert_eq!(Some(&0xff), vec.get(3));
        assert_eq!(Some(&0x00), vec.get(4));
        assert_eq!(Some(&0xff), vec.get(5));
        assert_eq!(Some(&0x00), vec.get(6));
        assert_eq!(Some(&0xff), vec.get(7));
    }

    #[test]
    fn test_small_vec_try_from_uint_should_failed() {
        let vec_result: Result<Vec<u8, 0>, _> = 1u8.try_into();

        assert!(vec_result.is_err());
    }
}
