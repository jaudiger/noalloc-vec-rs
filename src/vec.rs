use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use core::mem::size_of;
use core::ops::Deref;
use core::ops::DerefMut;
use core::ptr;
use core::slice;

use crate::assert_lte;

/// A fixed-size vector with a maximum length specified at compile time.
#[derive(Debug)]
pub struct Vec<T, const MAX_LENGTH: usize> {
    array: [MaybeUninit<T>; MAX_LENGTH],
    length: usize,
}

/// An iterator over the elements of a `Vec`.
pub struct IntoIter<T, const MAX_LENGTH: usize> {
    vec: Vec<T, MAX_LENGTH>,
    next: usize,
}

impl<T, const MAX_LENGTH: usize> IntoIter<T, MAX_LENGTH> {
    /// Creates a new `IntoIter` from `vec`.
    #[must_use]
    pub const fn new(vec: Vec<T, MAX_LENGTH>) -> Self {
        Self { vec, next: 0 }
    }
}

impl<T, const MAX_LENGTH: usize> Vec<T, MAX_LENGTH> {
    /// Creates a new, empty `Vec`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            array: [const { MaybeUninit::uninit() }; MAX_LENGTH],
            length: 0,
        }
    }

    /// Attempts to push `value` onto the vector.
    ///
    /// Returns `Ok(())` if successful, or `Err(value)` if the vector is full,
    /// allowing the caller to recover the rejected element.
    pub const fn push(&mut self, value: T) -> Result<(), T> {
        if self.length < MAX_LENGTH {
            self.push_unchecked(value);

            Ok(())
        } else {
            Err(value)
        }
    }

    /// Pushes `value` onto the vector without checking capacity.
    ///
    /// # Panics
    ///
    /// Panics if the vector is full.
    pub const fn push_unchecked(&mut self, value: T) {
        self.array[self.length].write(value);
        self.length += 1;
    }

    /// Removes and returns the last element, or `None` if the vector is empty.
    #[must_use]
    pub fn pop(&mut self) -> Option<T> {
        if self.length > 0 {
            // This is a safe operation because we've checked that the vector is not empty
            unsafe { Some(self.pop_unchecked()) }
        } else {
            None
        }
    }

    /// Removes and returns the last element without checking if the vector is empty.
    ///
    /// # Safety
    ///
    /// The vector must not be empty.
    #[must_use]
    pub unsafe fn pop_unchecked(&mut self) -> T {
        self.length -= 1;
        unsafe { self.take_unchecked(self.length) }
    }

    /// Writes `value` at `index` in the vector.
    ///
    /// Returns `Ok(())` if successful, or `Err(value)` if `index` is out of bounds,
    /// allowing the caller to recover the rejected element.
    pub const fn write(&mut self, index: usize, value: T) -> Result<(), T> {
        if index <= self.length && index < MAX_LENGTH {
            self.write_unchecked(index, value);

            Ok(())
        } else {
            Err(value)
        }
    }

    /// Writes `value` at `index` without checking bounds.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub const fn write_unchecked(&mut self, index: usize, value: T) {
        // Make sure all the previous bytes are initialized before reading the array
        self.array[index].write(value);
        if index >= self.length {
            self.length = index + 1;
        }
    }

    /// Writes `value` into the vector starting at `index`.
    ///
    /// Returns `Err(())` if `index + value.len()` exceeds `MAX_LENGTH`.
    #[allow(clippy::result_unit_err)]
    pub const fn write_slice(&mut self, index: usize, value: &[T]) -> Result<(), ()>
    where
        T: Copy,
    {
        if index <= self.length && index + value.len() <= MAX_LENGTH {
            self.write_slice_unchecked(index, value);

            Ok(())
        } else {
            Err(())
        }
    }

    /// Writes `value` into the vector starting at `start_index` without checking bounds.
    ///
    /// # Panics
    ///
    /// Panics if `start_index + value.len()` exceeds `MAX_LENGTH`.
    pub const fn write_slice_unchecked(&mut self, mut start_index: usize, value: &[T])
    where
        T: Copy,
    {
        // Make sure all the previous bytes are initialized before reading the array
        let mut index = 0;
        while index < value.len() {
            self.write_unchecked(start_index, value[index]);

            index += 1;
            start_index += 1;
        }
    }

    /// Inserts `value` at `index`, shifting subsequent elements to the right.
    ///
    /// Returns `Ok(())` if successful, or `Err(value)` if `index` is out of bounds
    /// or the vector is full, allowing the caller to recover the rejected element.
    pub const fn insert(&mut self, index: usize, value: T) -> Result<(), T> {
        // Check if the element can be inserted
        if index > self.length || self.length + 1 > MAX_LENGTH {
            Err(value)
        } else {
            // Shift all the elements after the index to the right
            unsafe {
                let base = self.array.as_mut_ptr().cast::<T>();
                let dst = base.add(index);
                ptr::copy(dst, dst.add(1), self.length - index);
                ptr::write(dst, value);
            }

            self.length += 1;

            Ok(())
        }
    }

    /// Removes and returns the element at `index`, or `None` if out of bounds.
    #[must_use]
    pub const fn remove(&mut self, index: usize) -> Option<T> {
        if index < self.length {
            // Shift all the elements after the index to the left
            let value = unsafe {
                let base = self.array.as_mut_ptr().cast::<T>();
                let src = base.add(index);
                let value = src.read();
                ptr::copy(src.add(1), src, self.length - index - 1);
                value
            };

            self.length -= 1;

            Some(value)
        } else {
            None
        }
    }

    /// Returns the element at `index` without checking bounds.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.length`.
    #[must_use]
    unsafe fn take_unchecked(&self, index: usize) -> T {
        unsafe { self.array.get_unchecked(index).as_ptr().read() }
    }

    /// Shortens the vector to `new_length`, dropping excess elements. Does nothing if `new_length >= self.length`.
    pub fn truncate(&mut self, new_length: usize) {
        if new_length >= self.length {
            return;
        }

        // Update the length
        let remaining_len = self.length - new_length;
        self.length = new_length;

        // Drop the old elements that are outside of the new length
        let start_slice = unsafe { self.as_mut_ptr().add(new_length) };
        let slice_to_drop = ptr::slice_from_raw_parts_mut(start_slice, remaining_len);
        unsafe {
            ptr::drop_in_place(slice_to_drop);
        }
    }

    /// Clears the vector, removing all elements.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Returns the contents as a slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.array.as_ptr().cast::<T>(), self.length) }
    }

    /// Returns the contents as a mutable slice.
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.array.as_mut_ptr().cast::<T>(), self.length) }
    }

    /// Moves all elements of `from_array` into a new `Vec`.
    ///
    /// # Safety
    ///
    /// `LENGTH` must not exceed `MAX_LENGTH`.
    #[must_use]
    unsafe fn from_array_unchecked<const LENGTH: usize>(from_array: [T; LENGTH]) -> Self {
        let mut vec = Self::new();

        // Do not drop the elements of the array, since we're moving them into the vector
        let array = ManuallyDrop::new(from_array);

        while vec.length < array.len() {
            vec.array[vec.length] =
                MaybeUninit::new(unsafe { ptr::read(&raw const array[vec.length]) });
            vec.length += 1;
        }

        vec
    }

    /// Copies all elements of `from_slice` into a new `Vec`.
    ///
    /// # Panics
    ///
    /// Panics if `from_slice.len()` exceeds `MAX_LENGTH`.
    #[must_use]
    const fn from_slice_unchecked(from_slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::new();

        while vec.length < from_slice.len() {
            vec.array[vec.length] = MaybeUninit::new(from_slice[vec.length]);
            vec.length += 1;
        }

        vec
    }

    /// Decodes `value` as a little-endian integer into at most `max_length` bytes, trimming trailing zeros.
    ///
    /// # Panics
    ///
    /// Panics if `max_length` exceeds `MAX_LENGTH`.
    #[must_use]
    fn from_uint_unchecked(mut value: u64, max_length: usize) -> Self
    where
        T: From<u8>,
    {
        let mut vec = Self::new();

        let mut real_length = 0;
        let mut index = 0;
        while index < max_length {
            let byte = (value & 0xff) as u8;
            if byte != 0 {
                real_length = index + 1;
            }

            vec.push_unchecked(byte.into());

            // Shift the value to the right
            value >>= 8;

            index += 1;
        }

        vec.length = real_length;
        vec
    }

    /// Encodes the elements as a little-endian `u64`, treating each element as a byte.
    #[must_use]
    fn to_uint(&self) -> u64
    where
        T: Into<u8>,
    {
        let mut value = 0;
        let mut index = 0;
        while index < self.len() {
            // This is a safe operation because we know that the index is within bounds
            let byte = unsafe { self.take_unchecked(index).into() };

            value |= u64::from(byte) << (index * 8);

            index += 1;
        }

        value
    }

    /// Returns the number of elements in the vector.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns the number of elements that can still be pushed before the vector is full.
    #[must_use]
    pub const fn remaining_len(&self) -> usize {
        MAX_LENGTH - self.length
    }

    /// Returns `true` if the vector contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Default implementation for `Vec`.
///
/// This allows creating a `Vec` using `Vec::default()`.
impl<T, const MAX_LENGTH: usize> Default for Vec<T, MAX_LENGTH> {
    /// Creates a new, empty `Vec`.
    fn default() -> Self {
        Self::new()
    }
}

/// Drop implementation for `Vec`.
///
/// This ensures that all elements in the `Vec` are properly dropped when the `Vec` goes out of scope.
impl<T, const LENGTH: usize> Drop for Vec<T, LENGTH> {
    /// Drops all elements in the `Vec`.
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
        }
    }
}

/// Implementation of `IntoIterator` for `&Vec`.
///
/// This allows iterating over references to the elements of a `Vec`.
impl<'a, T, const MAX_LENGTH: usize> IntoIterator for &'a Vec<T, MAX_LENGTH> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    /// Returns an iterator over the elements of the `Vec`.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Implementation of `Iterator` for `IntoIter`.
///
/// This allows iterating over the elements of a `Vec` by value.
impl<T, const MAX_LENGTH: usize> Iterator for IntoIter<T, MAX_LENGTH> {
    type Item = T;

    /// Returns the next element in the iterator.
    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.vec.len() {
            // This is a safe operation because we know that the index is within bounds
            let value = unsafe { self.vec.take_unchecked(self.next) };
            self.next += 1;

            Some(value)
        } else {
            None
        }
    }

    /// Returns the exact number of remaining elements as both the lower and upper bound.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.vec.length - self.next;
        (len, Some(len))
    }
}

impl<T, const MAX_LENGTH: usize> ExactSizeIterator for IntoIter<T, MAX_LENGTH> {}

/// Drop implementation for `IntoIter`.
///
/// This ensures that all remaining elements in the `IntoIter` are properly dropped when the `IntoIter` goes out of scope.
impl<T, const MAX_LENGTH: usize> Drop for IntoIter<T, MAX_LENGTH> {
    /// Drops all remaining elements in the `IntoIter`.
    fn drop(&mut self) {
        unsafe {
            // Drop all the remaining elements, and set the length to 0
            ptr::drop_in_place(&raw mut self.vec.as_mut_slice()[self.next..]);
            self.vec.length = 0;
        }
    }
}

/// Implementation of `IntoIterator` for `Vec`.
///
/// This allows converting a `Vec` into an `IntoIter`.
impl<T, const MAX_LENGTH: usize> IntoIterator for Vec<T, MAX_LENGTH> {
    type Item = T;
    type IntoIter = IntoIter<T, MAX_LENGTH>;

    /// Converts the `Vec` into an `IntoIter`.
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

/// Implementation of `PartialEq` for `Vec`.
///
/// This allows comparing two `Vec`s for equality.
impl<TA, TB, const MAX_LENGTH_A: usize, const MAX_LENGTH_B: usize> PartialEq<Vec<TB, MAX_LENGTH_B>>
    for Vec<TA, MAX_LENGTH_A>
where
    TA: PartialEq<TB>,
{
    /// Compares two `Vec`s for equality.
    fn eq(&self, other: &Vec<TB, MAX_LENGTH_B>) -> bool {
        <[TA]>::eq(self, &**other)
    }
}

/// Implementation of `Eq` for `Vec`.
///
/// This allows comparing two `Vec`s for equality.
impl<T, const MAX_LENGTH: usize> Eq for Vec<T, MAX_LENGTH> where T: Eq {}

/// Implementation of `TryFrom` for `Vec`.
///
/// This allows converting a slice into a `Vec`.
impl<T: Copy, const MAX_LENGTH: usize> TryFrom<&[T]> for Vec<T, MAX_LENGTH> {
    type Error = ();

    /// Converts `values` into a `Vec`, returning `Err(())` if the slice is longer than `MAX_LENGTH`.
    fn try_from(values: &[T]) -> Result<Self, Self::Error> {
        // Runtime check
        if values.len() > MAX_LENGTH {
            return Err(());
        }

        Ok(Self::from_slice_unchecked(values))
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a `Vec` into another `Vec`.
impl<T: Copy, const LENGTH: usize, const MAX_LENGTH: usize> From<&Vec<T, LENGTH>>
    for Vec<T, MAX_LENGTH>
{
    /// Converts a `Vec` into another `Vec`.
    fn from(values: &Vec<T, LENGTH>) -> Self {
        // Build time assertion
        assert_lte!(LENGTH, MAX_LENGTH);

        Self::from_slice_unchecked(values)
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting an array into a `Vec`.
impl<T, const LENGTH: usize, const MAX_LENGTH: usize> From<[T; LENGTH]> for Vec<T, MAX_LENGTH> {
    /// Converts an array into a `Vec`.
    fn from(values: [T; LENGTH]) -> Self {
        // Build time assertion
        assert_lte!(LENGTH, MAX_LENGTH);

        // This is a safe operation because we check at build time that the length is sufficient
        unsafe { Self::from_array_unchecked(values) }
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a reference to an array into a `Vec`.
impl<T: Copy, const LENGTH: usize, const MAX_LENGTH: usize> From<&[T; LENGTH]>
    for Vec<T, MAX_LENGTH>
{
    /// Converts a reference to an array into a `Vec`.
    fn from(values: &[T; LENGTH]) -> Self {
        // Build time assertion
        assert_lte!(LENGTH, MAX_LENGTH);

        Self::from_slice_unchecked(values)
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a `u8` into a `Vec`.
impl<const MAX_LENGTH: usize> From<u8> for Vec<u8, MAX_LENGTH> {
    /// Converts a `u8` into a `Vec`.
    fn from(value: u8) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u8>();
        assert_lte!(VALUE_LENGTH, MAX_LENGTH);

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a `u16` into a `Vec`.
impl<const MAX_LENGTH: usize> From<u16> for Vec<u8, MAX_LENGTH> {
    /// Converts a `u16` into a `Vec`.
    fn from(value: u16) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u16>();
        assert_lte!(VALUE_LENGTH, MAX_LENGTH);

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a `u32` into a `Vec`.
impl<const MAX_LENGTH: usize> From<u32> for Vec<u8, MAX_LENGTH> {
    /// Converts a `u32` into a `Vec`.
    fn from(value: u32) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u32>();
        assert_lte!(VALUE_LENGTH, MAX_LENGTH);

        Self::from_uint_unchecked(u64::from(value), VALUE_LENGTH)
    }
}

/// Implementation of `From` for `Vec`.
///
/// This allows converting a `u64` into a `Vec`.
impl<const MAX_LENGTH: usize> From<u64> for Vec<u8, MAX_LENGTH> {
    /// Converts a `u64` into a `Vec`.
    fn from(value: u64) -> Self {
        // Build time assertion
        const VALUE_LENGTH: usize = size_of::<u64>();
        assert_lte!(VALUE_LENGTH, MAX_LENGTH);

        Self::from_uint_unchecked(value, VALUE_LENGTH)
    }
}

/// Implementation of `Deref` for `Vec`.
///
/// This allows dereferencing a `Vec` to get a slice of its elements.
impl<T, const MAX_LENGTH: usize> Deref for Vec<T, MAX_LENGTH> {
    type Target = [T];

    /// Dereferences the `Vec` to get a slice of its elements.
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Implementation of `DerefMut` for `Vec`.
///
/// This allows dereferencing a mutable `Vec` to get a mutable slice of its elements.
impl<T, const MAX_LENGTH: usize> DerefMut for Vec<T, MAX_LENGTH> {
    /// Dereferences the mutable `Vec` to get a mutable slice of its elements.
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Implementation of `From` for `u8`.
///
/// This allows converting a `Vec` into a `u8`.
impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u8
where
    T: Into<Self>,
{
    #[allow(clippy::cast_possible_truncation)]
    /// Converts a `Vec` into a `u8`.
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

/// Implementation of `From` for `u16`.
///
/// This allows converting a `Vec` into a `u16`.
impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u16
where
    T: Into<u8>,
{
    #[allow(clippy::cast_possible_truncation)]
    /// Converts a `Vec` into a `u16`.
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

/// Implementation of `From` for `u32`.
///
/// This allows converting a `Vec` into a `u32`.
impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u32
where
    T: Into<u8>,
{
    #[allow(clippy::cast_possible_truncation)]
    /// Converts a `Vec` into a `u32`.
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

/// Implementation of `From` for `u64`.
///
/// This allows converting a `Vec` into a `u64`.
impl<T, const MAX_LENGTH: usize> From<&Vec<T, MAX_LENGTH>> for u64
where
    T: Into<u8>,
{
    /// Converts a `Vec` into a `u64`.
    fn from(value: &Vec<T, MAX_LENGTH>) -> Self {
        value.to_uint() as Self
    }
}

/// Implementation of `Extend` for `Vec`.
///
/// This allows extending a `Vec` with references to elements.
impl<'a, T, const MAX_LENGTH: usize> Extend<&'a T> for Vec<T, MAX_LENGTH>
where
    T: 'a + Clone,
{
    /// Extends the `Vec` with elements from `iter`.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields more elements than the remaining capacity.
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        for elem in iter.into_iter().cloned() {
            self.push_unchecked(elem);
        }
    }
}

/// Implementation of `Extend` for `Vec`.
///
/// This allows extending a `Vec` with elements.
impl<T, const MAX_LENGTH: usize> Extend<T> for Vec<T, MAX_LENGTH> {
    /// Extends the `Vec` with elements from `iter`.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields more elements than the remaining capacity.
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for elem in iter {
            self.push_unchecked(elem);
        }
    }
}

/// Implementation of `Clone` for `Vec`.
///
/// This allows cloning a `Vec`.
impl<T, const MAX_LENGTH: usize> Clone for Vec<T, MAX_LENGTH>
where
    T: Clone,
{
    /// Clones the `Vec`.
    fn clone(&self) -> Self {
        let mut new_vec = Self::new();
        for elem in self {
            new_vec.push_unchecked(elem.clone());
        }

        new_vec
    }
}

/// Implementation of `FromIterator` for `Vec`.
///
/// This allows constructing a `Vec` from an Iterator, eg. with
/// `Iterator::collect`.
impl<T, const MAX_LENGTH: usize> FromIterator<T> for Vec<T, MAX_LENGTH> {
    /// Collects an iterator into a `Vec`.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields more than `MAX_LENGTH` elements.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
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
        assert_eq!(Err(2), vec.push(2));
    }

    #[test]
    fn test_vec_push_unchecked() {
        let mut vec = Vec::<u8, 1>::new();

        vec.push_unchecked(1);

        assert_eq!(1, vec.len());
        assert!(!vec.is_empty());
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
    fn test_vec_pop_unchecked() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(1, unsafe { vec.pop_unchecked() });
        assert_eq!(0, vec.len());
        assert_eq!(None, vec.pop());
    }

    #[test]
    fn test_vec_write() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Ok(()), vec.write(0, 1));
        assert_eq!(1, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_write_out_of_bound() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Err(1), vec.write(3, 1));
    }

    #[test]
    fn test_vec_write_unchecked() {
        let mut vec = Vec::<u8, 3>::new();

        vec.write_unchecked(0, 1);

        assert_eq!(1, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_write_slice() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Ok(()), vec.write_slice(0, &[1, 2, 3]));
        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_write_slice_out_of_bound() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Err(()), vec.write_slice(1, &[1, 2, 3]));
    }

    #[test]
    fn test_vec_write_slice_unchecked() {
        let mut vec = Vec::<u8, 3>::new();

        vec.write_slice_unchecked(0, &[1, 2, 3]);

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_insert_at_start() {
        let mut vec = Vec::<u8, 3>::new();

        assert_eq!(Ok(()), vec.insert(0, 1));
        assert_eq!(1, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_insert_in_middle() {
        let mut vec = Vec::<u8, 4>::from([1, 2, 3]);

        assert_eq!(Ok(()), vec.insert(1, 4));
        assert_eq!(4, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&4), vec.get(1));
        assert_eq!(Some(&2), vec.get(2));
        assert_eq!(Some(&3), vec.get(3));
    }

    #[test]
    fn test_vec_insert_out_of_bound() {
        let mut vec = Vec::<u8, 0>::new();

        assert_eq!(Err(1), vec.insert(1, 1));
    }

    #[test]
    fn test_vec_remove() {
        let mut vec = Vec::<u8, 3>::from([1, 2, 3]);

        assert_eq!(Some(2), vec.remove(1));
        assert_eq!(2, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&3), vec.get(1));
        assert_eq!(None, vec.get(2));
    }

    #[test]
    fn test_vec_remove_last() {
        let mut vec = Vec::<u8, 3>::from([1, 2, 3]);

        assert_eq!(Some(3), vec.remove(2));
        assert_eq!(2, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(None, vec.get(2));
    }

    #[test]
    fn test_vec_remove_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(None, vec.remove(1));
    }

    #[test]
    fn test_vec_get() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(Some(&1), vec.first());
        assert_eq!(1, vec.len());
        assert_eq!(None, vec.get(1));
    }

    #[test]
    fn test_vec_get_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(None, vec.get(1));
        assert_eq!(1, vec.len());
    }

    #[test]
    fn test_vec_get_unchecked() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(&1, unsafe { vec.get_unchecked(0) });
        assert_eq!(1, vec.len());
    }

    #[test]
    fn test_vec_get_mut() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(Some(&mut 1), vec.get_mut(0));
        assert_eq!(1, vec.len());
        assert_eq!(None, vec.get_mut(1));
    }

    #[test]
    fn test_vec_get_mut_out_of_bound() {
        let mut vec = Vec::<u8, 1>::new();
        let _ = vec.push(1);

        assert_eq!(None, vec.get_mut(1));
        assert_eq!(1, vec.len());
    }

    #[test]
    fn test_vec_extend() {
        let mut vec = Vec::<u8, 3>::new();
        let array: [u8; 3] = [1, 2, 3];

        vec.extend(array);
        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_as_slice_with_empty_vec() {
        let vec = Vec::<u8, 1>::new();

        let array = vec.as_slice();

        assert_eq!(0, array.len());
    }

    #[test]
    fn test_as_mut_slice_with_empty_vec() {
        let mut vec = Vec::<u8, 1>::new();

        let array = vec.as_mut_slice();

        assert_eq!(0, array.len());
    }

    #[test]
    fn test_vec_truncate() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert!(!vec.is_empty());

        vec.truncate(0);

        assert_eq!(0, vec.len());
        assert!(vec.is_empty());
    }

    #[test]
    fn test_vec_truncate_with_new_length_equal_current_length() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert!(!vec.is_empty());

        vec.truncate(1);

        assert_eq!(1, vec.len());
        assert!(!vec.is_empty());
    }

    #[test]
    fn test_vec_truncate_with_new_length_superior_to_current_length() {
        let mut vec = Vec::<u8, 1>::new();

        assert_eq!(Ok(()), vec.push(1));
        assert!(!vec.is_empty());

        vec.truncate(2);

        assert_eq!(1, vec.len());
        assert!(!vec.is_empty());
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
    fn test_vec_try_from_array_as_slice() {
        let vec: Vec<u8, 3> = [1, 2, 3].as_slice().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_try_from_array_as_slice_shorter_than_vec_size() {
        let vec: Vec<u8, 8> = [1, 2, 3].as_slice().try_into().unwrap();

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
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
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_from_array_same_size_as_vec() {
        let vec: Vec<u8, 3> = [1, 2, 3].into();

        assert_eq!(3, vec.len());
        assert_eq!(Some(&1), vec.first());
        assert_eq!(Some(&2), vec.get(1));
        assert_eq!(Some(&3), vec.get(2));
        assert_eq!(None, vec.get(3));
    }

    #[test]
    fn test_vec_from_u8() {
        let vec: Vec<u8, 8> = 0xffu8.into();

        assert_eq!(1, vec.len());
        assert_eq!(Some(&0xff), vec.first());
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
        assert_eq!(Some(&0x00), vec.first());
        assert_eq!(Some(&0xff), vec.get(1));
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

        assert_eq!(1, vec.len());
        assert_eq!(Some(&0xff), vec.first());
        assert_eq!(None, vec.get(1));
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
        assert_eq!(Some(&0x00), vec.first());
        assert_eq!(Some(&0xff), vec.get(1));
        assert_eq!(Some(&0x00), vec.get(2));
        assert_eq!(Some(&0xff), vec.get(3));
        assert_eq!(None, vec.get(4));
        assert_eq!(None, vec.get(5));
        assert_eq!(None, vec.get(6));
        assert_eq!(None, vec.get(7));
    }

    #[test]
    fn test_vec_from_u64() {
        let vec: Vec<u8, 8> = 0xff00_ff00_ff00_ff00_u64.into();

        assert_eq!(8, vec.len());
        assert_eq!(Some(&0x00), vec.first());
        assert_eq!(Some(&0xff), vec.get(1));
        assert_eq!(Some(&0x00), vec.get(2));
        assert_eq!(Some(&0xff), vec.get(3));
        assert_eq!(Some(&0x00), vec.get(4));
        assert_eq!(Some(&0xff), vec.get(5));
        assert_eq!(Some(&0x00), vec.get(6));
        assert_eq!(Some(&0xff), vec.get(7));
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
    #[allow(unused_mut)]
    fn test_deref_mut_with_empty_vec() {
        let mut vec = Vec::<u8, 1>::new();

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

    #[test]
    fn test_from_iter() {
        let vec: Vec<u8, 5> = [1, 2, 3].into_iter().collect();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[2], 3);
    }
}
