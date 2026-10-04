//! Columnar arrays: the physical representation every operator works on.
//!
//! An [`Array`] holds one column's worth of values for a batch of rows, stored
//! as a contiguous buffer plus a separate validity bitmap. Nulls are tracked
//! out-of-band rather than as sentinel values, which means:
//!
//! * the value buffer stays densely packed and branch-free to iterate, and
//! * a column that happens to contain no nulls carries no bitmap at all, so the
//!   common case costs nothing.
//!
//! This is the representation that makes vectorized execution worthwhile: a
//! filter over a million rows becomes a tight loop over a `&[i64]`, not a
//! million enum matches.

use crate::error::{Error, Result};
use crate::types::{DataType, Value};
use std::sync::Arc;

/// A bitmap marking which slots in an array hold a non-null value.
///
/// Stored as one bit per row, least-significant bit first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmap {
    bits: Vec<u8>,
    len: usize,
}

impl Bitmap {
    /// Creates a bitmap of `len` slots, all valid.
    pub fn all_valid(len: usize) -> Self {
        Bitmap { bits: vec![0xFF; len.div_ceil(8)], len }
    }

    /// Creates a bitmap of `len` slots, all null.
    pub fn all_null(len: usize) -> Self {
        Bitmap { bits: vec![0x00; len.div_ceil(8)], len }
    }

    /// Builds a bitmap from an iterator of validity flags.
    ///
    /// Named `collect_from` rather than implementing `FromIterator` because the
    /// bitmap is built from flags, not from the array elements it describes;
    /// `.collect()` on a column would read as the wrong operation.
    pub fn collect_from(iter: impl IntoIterator<Item = bool>) -> Self {
        let mut bits = Vec::new();
        let mut len = 0usize;
        for (i, v) in iter.into_iter().enumerate() {
            if i % 8 == 0 {
                bits.push(0);
            }
            if v {
                bits[i / 8] |= 1 << (i % 8);
            }
            len += 1;
        }
        Bitmap { bits, len }
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the bitmap covers no slots.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether slot `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        self.bits[i / 8] & (1 << (i % 8)) != 0
    }

    /// Marks slot `i` as null.
    pub fn set_null(&mut self, i: usize) {
        self.bits[i / 8] &= !(1 << (i % 8));
    }

    /// Counts the null slots.
    pub fn null_count(&self) -> usize {
        (0..self.len).filter(|&i| !self.is_valid(i)).count()
    }

    /// Returns the bitmap restricted to the given row positions.
    pub fn take(&self, indices: &[usize]) -> Bitmap {
        Bitmap::collect_from(indices.iter().map(|&i| self.is_valid(i)))
    }

    /// Logical AND of two bitmaps of equal length.
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        debug_assert_eq!(self.len, other.len);
        let bits = self.bits.iter().zip(other.bits.iter()).map(|(a, b)| a & b).collect();
        Bitmap { bits, len: self.len }
    }
}

/// The typed storage backing an [`Array`].
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    /// 64-bit integers.
    Int64(Vec<i64>),
    /// 64-bit floats.
    Float64(Vec<f64>),
    /// UTF-8 strings.
    Utf8(Vec<String>),
    /// Booleans.
    Boolean(Vec<bool>),
    /// Dates as days since the epoch.
    Date32(Vec<i32>),
    /// A column that is entirely NULL, carrying only its length.
    Null(usize),
}

/// One column of a record batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: ArrayData,
    /// `None` means every slot is valid -- the fast path.
    validity: Option<Bitmap>,
}

/// A reference-counted array. Operators clone these freely; the underlying
/// buffers are shared, so projecting or reordering columns is pointer work.
pub type ArrayRef = Arc<Array>;

impl Array {
    /// Wraps typed data with no nulls.
    pub fn new(data: ArrayData) -> Self {
        Array { data, validity: None }
    }

    /// Wraps typed data with an explicit validity bitmap.
    pub fn with_validity(data: ArrayData, validity: Option<Bitmap>) -> Self {
        // A bitmap with no nulls is pure overhead on every access; drop it.
        let validity = match validity {
            Some(b) if b.null_count() == 0 => None,
            other => other,
        };
        Array { data, validity }
    }

    /// Builds an array from scalar values, inferring the type from the first
    /// non-null element.
    pub fn from_values(values: Vec<Value>, data_type: DataType) -> Result<Self> {
        let len = values.len();
        let mut validity = Bitmap::all_valid(len);
        let mut any_null = false;

        macro_rules! build {
            ($variant:ident, $default:expr, $extract:pat => $val:expr) => {{
                let mut buf = Vec::with_capacity(len);
                for (i, v) in values.into_iter().enumerate() {
                    match v {
                        Value::Null => {
                            validity.set_null(i);
                            any_null = true;
                            buf.push($default);
                        }
                        $extract => buf.push($val),
                        other => {
                            return Err(Error::typ(format!(
                                "expected {data_type} but found {} in column data",
                                other.data_type()
                            )))
                        }
                    }
                }
                ArrayData::$variant(buf)
            }};
        }

        let data = match data_type {
            DataType::Int64 => build!(Int64, 0i64, Value::Int64(v) => v),
            DataType::Float64 => build!(Float64, 0f64, Value::Float64(v) => v),
            DataType::Utf8 => build!(Utf8, String::new(), Value::Utf8(v) => v),
            DataType::Boolean => build!(Boolean, false, Value::Boolean(v) => v),
            DataType::Date32 => build!(Date32, 0i32, Value::Date32(v) => v),
            DataType::Null => ArrayData::Null(len),
        };

        Ok(Array::with_validity(data, if any_null { Some(validity) } else { None }))
    }

    /// The array's logical type.
    pub fn data_type(&self) -> DataType {
        match &self.data {
            ArrayData::Int64(_) => DataType::Int64,
            ArrayData::Float64(_) => DataType::Float64,
            ArrayData::Utf8(_) => DataType::Utf8,
            ArrayData::Boolean(_) => DataType::Boolean,
            ArrayData::Date32(_) => DataType::Date32,
            ArrayData::Null(_) => DataType::Null,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match &self.data {
            ArrayData::Int64(v) => v.len(),
            ArrayData::Float64(v) => v.len(),
            ArrayData::Utf8(v) => v.len(),
            ArrayData::Boolean(v) => v.len(),
            ArrayData::Date32(v) => v.len(),
            ArrayData::Null(n) => *n,
        }
    }

    /// Whether the array has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The underlying typed storage.
    pub fn data(&self) -> &ArrayData {
        &self.data
    }

    /// The validity bitmap, if any nulls are present.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether row `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        if matches!(self.data, ArrayData::Null(_)) {
            return false;
        }
        match &self.validity {
            None => true,
            Some(b) => b.is_valid(i),
        }
    }

    /// Number of NULLs.
    pub fn null_count(&self) -> usize {
        if let ArrayData::Null(n) = self.data {
            return n;
        }
        self.validity.as_ref().map_or(0, |b| b.null_count())
    }

    /// Reads row `i` as a scalar.
    ///
    /// This is the slow path, used for result materialization and grouping
    /// keys. Hot loops use the typed slice accessors below instead.
    pub fn value(&self, i: usize) -> Value {
        if !self.is_valid(i) {
            return Value::Null;
        }
        match &self.data {
            ArrayData::Int64(v) => Value::Int64(v[i]),
            ArrayData::Float64(v) => Value::Float64(v[i]),
            ArrayData::Utf8(v) => Value::Utf8(v[i].clone()),
            ArrayData::Boolean(v) => Value::Boolean(v[i]),
            ArrayData::Date32(v) => Value::Date32(v[i]),
            ArrayData::Null(_) => Value::Null,
        }
    }

    /// The integer slice, if this is an `Int64` array.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.data {
            ArrayData::Int64(v) => Some(v),
            _ => None,
        }
    }

    /// The float slice, if this is a `Float64` array.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match &self.data {
            ArrayData::Float64(v) => Some(v),
            _ => None,
        }
    }

    /// The string slice, if this is a `Utf8` array.
    pub fn as_str(&self) -> Option<&[String]> {
        match &self.data {
            ArrayData::Utf8(v) => Some(v),
            _ => None,
        }
    }

    /// The boolean slice, if this is a `Boolean` array.
    pub fn as_bool(&self) -> Option<&[bool]> {
        match &self.data {
            ArrayData::Boolean(v) => Some(v),
            _ => None,
        }
    }

    /// The date slice, if this is a `Date32` array.
    pub fn as_date32(&self) -> Option<&[i32]> {
        match &self.data {
            ArrayData::Date32(v) => Some(v),
            _ => None,
        }
    }

    /// Gathers the rows at `indices` into a new array.
    ///
    /// `take` is the single primitive behind filtering, sorting and the probe
    /// side of a hash join. Expressing all three as "compute positions, then
    /// gather" keeps the expensive part -- the actual data movement -- in one
    /// place and type-specialized.
    pub fn take(&self, indices: &[usize]) -> Array {
        macro_rules! gather {
            ($v:expr, $variant:ident) => {{
                let buf: Vec<_> = indices.iter().map(|&i| $v[i].clone()).collect();
                ArrayData::$variant(buf)
            }};
        }
        let data = match &self.data {
            ArrayData::Int64(v) => gather!(v, Int64),
            ArrayData::Float64(v) => gather!(v, Float64),
            ArrayData::Utf8(v) => gather!(v, Utf8),
            ArrayData::Boolean(v) => gather!(v, Boolean),
            ArrayData::Date32(v) => gather!(v, Date32),
            ArrayData::Null(_) => ArrayData::Null(indices.len()),
        };
        let validity = self.validity.as_ref().map(|b| b.take(indices));
        Array::with_validity(data, validity)
    }

    /// Returns rows `[offset, offset + length)`.
    pub fn slice(&self, offset: usize, length: usize) -> Array {
        let end = (offset + length).min(self.len());
        let indices: Vec<usize> = (offset.min(end)..end).collect();
        self.take(&indices)
    }

    /// Appends `other` to `self`, which must have the same type.
    pub fn concat(&self, other: &Array) -> Result<Array> {
        // A fully-null column adopts the other side's type, so that an empty
        // or all-null batch does not poison a concatenation.
        if matches!(self.data, ArrayData::Null(_)) && self.is_empty() {
            return Ok(other.clone());
        }
        if matches!(other.data, ArrayData::Null(_)) && other.is_empty() {
            return Ok(self.clone());
        }

        let validity = match (&self.validity, &other.validity) {
            (None, None) => None,
            _ => {
                let mut flags = Vec::with_capacity(self.len() + other.len());
                flags.extend((0..self.len()).map(|i| self.is_valid(i)));
                flags.extend((0..other.len()).map(|i| other.is_valid(i)));
                Some(Bitmap::collect_from(flags))
            }
        };

        macro_rules! join {
            ($a:expr, $b:expr, $variant:ident) => {{
                let mut buf = $a.clone();
                buf.extend($b.iter().cloned());
                ArrayData::$variant(buf)
            }};
        }
        let data = match (&self.data, &other.data) {
            (ArrayData::Int64(a), ArrayData::Int64(b)) => join!(a, b, Int64),
            (ArrayData::Float64(a), ArrayData::Float64(b)) => join!(a, b, Float64),
            (ArrayData::Utf8(a), ArrayData::Utf8(b)) => join!(a, b, Utf8),
            (ArrayData::Boolean(a), ArrayData::Boolean(b)) => join!(a, b, Boolean),
            (ArrayData::Date32(a), ArrayData::Date32(b)) => join!(a, b, Date32),
            (ArrayData::Null(a), ArrayData::Null(b)) => ArrayData::Null(a + b),
            (a, b) => {
                return Err(Error::typ(format!(
                    "cannot concatenate arrays of different types: {:?} and {:?}",
                    discriminant_name(a),
                    discriminant_name(b)
                )))
            }
        };
        Ok(Array::with_validity(data, validity))
    }

    /// Builds an array of `len` NULLs with the given type.
    pub fn new_null(data_type: DataType, len: usize) -> Array {
        let data = match data_type {
            DataType::Int64 => ArrayData::Int64(vec![0; len]),
            DataType::Float64 => ArrayData::Float64(vec![0.0; len]),
            DataType::Utf8 => ArrayData::Utf8(vec![String::new(); len]),
            DataType::Boolean => ArrayData::Boolean(vec![false; len]),
            DataType::Date32 => ArrayData::Date32(vec![0; len]),
            DataType::Null => ArrayData::Null(len),
        };
        Array { data, validity: Some(Bitmap::all_null(len)) }
    }
}

fn discriminant_name(d: &ArrayData) -> &'static str {
    match d {
        ArrayData::Int64(_) => "Int64",
        ArrayData::Float64(_) => "Float64",
        ArrayData::Utf8(_) => "Utf8",
        ArrayData::Boolean(_) => "Boolean",
        ArrayData::Date32(_) => "Date32",
        ArrayData::Null(_) => "Null",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmap_tracks_validity_per_slot() {
        let b = Bitmap::collect_from([true, false, true, true, false, false, true, false, true]);
        assert_eq!(b.len(), 9);
        assert_eq!(b.null_count(), 4);
        for (i, want) in
            [true, false, true, true, false, false, true, false, true].iter().enumerate()
        {
            assert_eq!(b.is_valid(i), *want, "slot {i}");
        }
    }

    #[test]
    fn bitmap_and_intersects_validity() {
        let a = Bitmap::collect_from([true, true, false, false]);
        let b = Bitmap::collect_from([true, false, true, false]);
        let c = a.and(&b);
        assert_eq!(
            (0..4).map(|i| c.is_valid(i)).collect::<Vec<_>>(),
            vec![true, false, false, false]
        );
    }

    #[test]
    fn a_bitmap_with_no_nulls_is_dropped() {
        // Carrying a bitmap that marks nothing costs a branch on every access
        // for no information, so the constructor discards it.
        let a = Array::with_validity(ArrayData::Int64(vec![1, 2, 3]), Some(Bitmap::all_valid(3)));
        assert!(a.validity().is_none());
        assert_eq!(a.null_count(), 0);
    }

    #[test]
    fn from_values_separates_nulls_from_data() {
        let a = Array::from_values(
            vec![Value::Int64(1), Value::Null, Value::Int64(3)],
            DataType::Int64,
        )
        .unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        assert!(!a.is_valid(1));
        assert_eq!(a.value(0), Value::Int64(1));
        assert_eq!(a.value(1), Value::Null);
        assert_eq!(a.value(2), Value::Int64(3));
        // The value buffer stays dense: the null slot holds a placeholder, so
        // the slice can still be iterated without branching.
        assert_eq!(a.as_i64().unwrap().len(), 3);
    }

    #[test]
    fn from_values_rejects_a_type_mismatch() {
        let err =
            Array::from_values(vec![Value::Int64(1), Value::Utf8("x".into())], DataType::Int64);
        assert!(err.is_err(), "a string in an INT64 column must be rejected");
    }

    #[test]
    fn take_gathers_rows_and_their_validity() {
        let a = Array::from_values(
            vec![Value::Utf8("a".into()), Value::Null, Value::Utf8("c".into())],
            DataType::Utf8,
        )
        .unwrap();
        let t = a.take(&[2, 1, 0, 2]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.value(0), Value::Utf8("c".into()));
        assert_eq!(t.value(1), Value::Null);
        assert_eq!(t.value(3), Value::Utf8("c".into()));
        assert_eq!(t.null_count(), 1);
    }

    #[test]
    fn concat_preserves_values_and_nulls() {
        let a = Array::from_values(vec![Value::Int64(1), Value::Null], DataType::Int64).unwrap();
        let b = Array::from_values(vec![Value::Int64(3)], DataType::Int64).unwrap();
        let c = a.concat(&b).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.value(0), Value::Int64(1));
        assert_eq!(c.value(1), Value::Null);
        assert_eq!(c.value(2), Value::Int64(3));
    }

    #[test]
    fn concat_rejects_mismatched_types() {
        let a = Array::new(ArrayData::Int64(vec![1]));
        let b = Array::new(ArrayData::Utf8(vec!["x".into()]));
        assert!(a.concat(&b).is_err());
    }

    #[test]
    fn slice_returns_a_window() {
        let a = Array::new(ArrayData::Int64((0..10).collect()));
        let s = a.slice(3, 4);
        assert_eq!(s.len(), 4);
        assert_eq!(s.as_i64().unwrap(), &[3, 4, 5, 6]);
        // Slicing past the end clamps rather than panicking.
        assert_eq!(a.slice(8, 100).len(), 2);
        assert_eq!(a.slice(100, 5).len(), 0);
    }

    #[test]
    fn an_all_null_array_reports_every_slot_null() {
        let a = Array::new_null(DataType::Float64, 5);
        assert_eq!(a.len(), 5);
        assert_eq!(a.null_count(), 5);
        assert!((0..5).all(|i| a.value(i) == Value::Null));
    }
}
