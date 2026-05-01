use crate::core::{FeaturizerError, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const HEADER_BYTES: usize = 128;
const DEFAULT_MAT_KEYS: [&str; 6] = [
    "instance_map",
    "inst_map",
    "instances",
    "labels",
    "segmentation",
    "mask",
];

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;

const MX_LOGICAL_CLASS: u8 = 3;

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug)]
struct Element<'a> {
    data_type: u32,
    payload: &'a [u8],
    total_size: usize,
}

#[derive(Debug)]
struct MatArray {
    name: String,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

pub fn load_instance_map(
    path: &Path,
    preferred_key: Option<&str>,
) -> Result<(Array2<u32>, String)> {
    let bytes = fs::read(path)?;
    let endian = detect_endian(&bytes, path)?;
    let arrays = parse_numeric_arrays(&bytes, endian, path)?;
    if arrays.is_empty() {
        return Err(FeaturizerError::InvalidInput(format!(
            "no numeric arrays found in MAT file '{}'",
            path.display()
        )));
    }

    if let Some(key) = preferred_key {
        let arr = arrays.get(key).ok_or_else(|| {
            FeaturizerError::InvalidInput(format!("key '{key}' not found in '{}'", path.display()))
        })?;
        let coerced = coerce_instance_map(arr)?;
        return Ok((coerced, key.to_string()));
    }

    for key in DEFAULT_MAT_KEYS {
        if let Some(arr) = arrays.get(key) {
            if let Ok(coerced) = coerce_instance_map(arr) {
                return Ok((coerced, key.to_string()));
            }
        }
    }

    let mut best_key: Option<String> = None;
    let mut best_map: Option<Array2<u32>> = None;
    let mut best_unique = 0usize;
    for (key, array) in &arrays {
        let Ok(candidate) = coerce_instance_map(array) else {
            continue;
        };
        let mut uniques: Vec<u32> = candidate.iter().copied().collect();
        uniques.sort_unstable();
        uniques.dedup();
        if uniques.len() > best_unique {
            best_unique = uniques.len();
            best_key = Some(key.clone());
            best_map = Some(candidate);
        }
    }

    match (best_map, best_key) {
        (Some(map), Some(key)) => Ok((map, key)),
        _ => Err(FeaturizerError::InvalidInput(format!(
            "no valid 2D instance map found in '{}'",
            path.display()
        ))),
    }
}

fn detect_endian(bytes: &[u8], path: &Path) -> Result<Endian> {
    if bytes.len() < HEADER_BYTES {
        return Err(FeaturizerError::InvalidInput(format!(
            "MAT file '{}' is smaller than 128-byte header",
            path.display()
        )));
    }
    match &bytes[126..128] {
        b"IM" => Ok(Endian::Little),
        b"MI" => Ok(Endian::Big),
        _ => Err(FeaturizerError::InvalidInput(format!(
            "MAT file '{}' has unsupported endian marker {:?}",
            path.display(),
            &bytes[126..128]
        ))),
    }
}

fn parse_numeric_arrays(
    bytes: &[u8],
    endian: Endian,
    path: &Path,
) -> Result<HashMap<String, MatArray>> {
    let mut offset = HEADER_BYTES;
    let mut out: HashMap<String, MatArray> = HashMap::new();
    while offset + 8 <= bytes.len() {
        let elem = parse_element(&bytes[offset..], endian)?;
        if elem.total_size == 0 {
            break;
        }
        if elem.data_type == MI_MATRIX {
            if let Some(array) = parse_matrix_element(elem.payload, endian)? {
                out.insert(array.name.clone(), array);
            }
        }
        offset += elem.total_size;
    }
    if offset > bytes.len() {
        return Err(FeaturizerError::InvalidInput(format!(
            "MAT parsing overflow in '{}'",
            path.display()
        )));
    }
    Ok(out)
}

fn parse_matrix_element(payload: &[u8], endian: Endian) -> Result<Option<MatArray>> {
    let mut inner_offset = 0usize;

    let flags = parse_element(&payload[inner_offset..], endian)?;
    inner_offset += flags.total_size;
    if flags.payload.len() < 8 {
        return Ok(None);
    }
    let class_bits = read_u32(&flags.payload[0..4], endian)?;
    let class = (class_bits & 0xFF) as u8;

    let dims = parse_element(&payload[inner_offset..], endian)?;
    inner_offset += dims.total_size;
    let (rows, cols) = parse_dims(dims.payload, endian)?;

    let name_elem = parse_element(&payload[inner_offset..], endian)?;
    inner_offset += name_elem.total_size;
    let name = parse_name(name_elem.payload)?;

    let real = parse_element(&payload[inner_offset..], endian)?;
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| FeaturizerError::InvalidInput("matrix shape overflow".to_string()))?;
    let values = parse_numeric_values(real.data_type, real.payload, expected_len, endian, class)?;
    Ok(Some(MatArray {
        name,
        rows,
        cols,
        values,
    }))
}

fn parse_element(bytes: &[u8], endian: Endian) -> Result<Element<'_>> {
    if bytes.len() < 8 {
        return Err(FeaturizerError::InvalidInput(
            "truncated MAT data element tag".to_string(),
        ));
    }

    let tag_word = read_u32(&bytes[0..4], endian)?;
    let (data_type, payload_size, header_size) = if is_small_element(tag_word, endian) {
        let dt = if matches!(endian, Endian::Little) {
            tag_word & 0xFFFF
        } else {
            tag_word >> 16
        };
        let sz = if matches!(endian, Endian::Little) {
            (tag_word >> 16) & 0xFFFF
        } else {
            tag_word & 0xFFFF
        };
        (dt, sz as usize, 4usize)
    } else {
        let dt = tag_word;
        let sz = read_u32(&bytes[4..8], endian)? as usize;
        (dt, sz, 8usize)
    };

    let padded_payload_size = align8(payload_size);
    let total_size = header_size
        .checked_add(padded_payload_size)
        .ok_or_else(|| FeaturizerError::InvalidInput("MAT element size overflow".to_string()))?;
    if bytes.len() < total_size {
        return Err(FeaturizerError::InvalidInput(
            "truncated MAT element payload".to_string(),
        ));
    }

    let payload_start = header_size;
    let payload_end = payload_start + payload_size;
    Ok(Element {
        data_type,
        payload: &bytes[payload_start..payload_end],
        total_size,
    })
}

fn is_small_element(tag_word: u32, endian: Endian) -> bool {
    match endian {
        Endian::Little => (tag_word >> 16) > 0,
        Endian::Big => (tag_word & 0xFFFF) > 0,
    }
}

fn parse_dims(payload: &[u8], endian: Endian) -> Result<(usize, usize)> {
    if payload.len() < 8 {
        return Err(FeaturizerError::InvalidInput(
            "MAT dimensions payload too short".to_string(),
        ));
    }
    let rows = read_i32(&payload[0..4], endian)?;
    let cols = read_i32(&payload[4..8], endian)?;
    if rows <= 0 || cols <= 0 {
        return Err(FeaturizerError::InvalidInput(format!(
            "invalid matrix shape ({rows}, {cols})"
        )));
    }
    Ok((rows as usize, cols as usize))
}

fn parse_name(payload: &[u8]) -> Result<String> {
    let trimmed_end = payload
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(payload.len());
    let raw = &payload[..trimmed_end];
    String::from_utf8(raw.to_vec())
        .map_err(|_| FeaturizerError::InvalidInput("MAT variable name is not UTF-8".to_string()))
}

fn parse_numeric_values(
    data_type: u32,
    payload: &[u8],
    expected_len: usize,
    endian: Endian,
    class: u8,
) -> Result<Vec<f64>> {
    let item_size = item_size_for_type(data_type)?;
    let expected_bytes = expected_len
        .checked_mul(item_size)
        .ok_or_else(|| FeaturizerError::InvalidInput("MAT array byte-size overflow".to_string()))?;
    if payload.len() < expected_bytes {
        return Err(FeaturizerError::InvalidInput(format!(
            "MAT array payload too short: need {expected_bytes}, got {}",
            payload.len()
        )));
    }

    if class == MX_LOGICAL_CLASS && data_type == MI_UINT8 {
        let mut out = Vec::with_capacity(expected_len);
        for b in payload.iter().take(expected_len) {
            out.push(if *b == 0 { 0.0 } else { 1.0 });
        }
        return Ok(out);
    }

    let mut out = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let offset = i * item_size;
        let chunk = &payload[offset..offset + item_size];
        let val = match data_type {
            MI_INT8 => i8::from_ne_bytes([chunk[0]]) as f64,
            MI_UINT8 => chunk[0] as f64,
            MI_INT16 => read_i16(chunk, endian)? as f64,
            MI_UINT16 => read_u16(chunk, endian)? as f64,
            MI_INT32 => read_i32(chunk, endian)? as f64,
            MI_UINT32 => read_u32(chunk, endian)? as f64,
            MI_SINGLE => read_f32(chunk, endian)? as f64,
            MI_DOUBLE => read_f64(chunk, endian)?,
            MI_INT64 => read_i64(chunk, endian)? as f64,
            MI_UINT64 => read_u64(chunk, endian)? as f64,
            _ => {
                return Err(FeaturizerError::InvalidInput(format!(
                    "unsupported MAT numeric data type {data_type}"
                )));
            }
        };
        out.push(val);
    }
    Ok(out)
}

fn coerce_instance_map(array: &MatArray) -> Result<Array2<u32>> {
    if array.rows == 0 || array.cols == 0 {
        return Err(FeaturizerError::InvalidInput(
            "instance map cannot be empty".to_string(),
        ));
    }

    let mut out = Array2::<u32>::zeros((array.rows, array.cols));
    for col in 0..array.cols {
        for row in 0..array.rows {
            let src_idx = row + col * array.rows;
            let value = array.values[src_idx];
            if !value.is_finite() {
                return Err(FeaturizerError::InvalidInput(
                    "instance map contains non-finite values".to_string(),
                ));
            }
            let rounded = value.round();
            if (value - rounded).abs() > 1e-6 {
                return Err(FeaturizerError::InvalidInput(
                    "instance map contains non-integral floating values".to_string(),
                ));
            }
            if rounded < 0.0 || rounded > u32::MAX as f64 {
                return Err(FeaturizerError::InvalidInput(
                    "instance map contains values outside uint32 range".to_string(),
                ));
            }
            out[[row, col]] = rounded as u32;
        }
    }
    Ok(out)
}

fn item_size_for_type(data_type: u32) -> Result<usize> {
    match data_type {
        MI_INT8 | MI_UINT8 => Ok(1),
        MI_INT16 | MI_UINT16 => Ok(2),
        MI_INT32 | MI_UINT32 | MI_SINGLE => Ok(4),
        MI_DOUBLE | MI_INT64 | MI_UINT64 => Ok(8),
        _ => Err(FeaturizerError::InvalidInput(format!(
            "unsupported MAT data type {data_type}"
        ))),
    }
}

fn align8(len: usize) -> usize {
    (len + 7) & !7
}

fn read_u16(bytes: &[u8], endian: Endian) -> Result<u16> {
    let arr: [u8; 2] = bytes
        .try_into()
        .map_err(|_| FeaturizerError::InvalidInput("truncated u16".to_string()))?;
    Ok(match endian {
        Endian::Little => u16::from_le_bytes(arr),
        Endian::Big => u16::from_be_bytes(arr),
    })
}

fn read_i16(bytes: &[u8], endian: Endian) -> Result<i16> {
    Ok(read_u16(bytes, endian)? as i16)
}

fn read_u32(bytes: &[u8], endian: Endian) -> Result<u32> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| FeaturizerError::InvalidInput("truncated u32".to_string()))?;
    Ok(match endian {
        Endian::Little => u32::from_le_bytes(arr),
        Endian::Big => u32::from_be_bytes(arr),
    })
}

fn read_i32(bytes: &[u8], endian: Endian) -> Result<i32> {
    Ok(read_u32(bytes, endian)? as i32)
}

fn read_u64(bytes: &[u8], endian: Endian) -> Result<u64> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| FeaturizerError::InvalidInput("truncated u64".to_string()))?;
    Ok(match endian {
        Endian::Little => u64::from_le_bytes(arr),
        Endian::Big => u64::from_be_bytes(arr),
    })
}

fn read_i64(bytes: &[u8], endian: Endian) -> Result<i64> {
    Ok(read_u64(bytes, endian)? as i64)
}

fn read_f32(bytes: &[u8], endian: Endian) -> Result<f32> {
    Ok(f32::from_bits(read_u32(bytes, endian)?))
}

fn read_f64(bytes: &[u8], endian: Endian) -> Result<f64> {
    Ok(f64::from_bits(read_u64(bytes, endian)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn append_tagged(buf: &mut Vec<u8>, data_type: u32, payload: &[u8]) {
        buf.extend_from_slice(&data_type.to_le_bytes());
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload);
        while !buf.len().is_multiple_of(8) {
            buf.push(0);
        }
    }

    fn make_numeric_matrix(name: &str, rows: i32, cols: i32, values: &[i32]) -> Vec<u8> {
        let mut matrix_payload: Vec<u8> = Vec::new();

        let mut flags = Vec::new();
        flags.extend_from_slice(&(MI_UINT32).to_le_bytes());
        flags.extend_from_slice(&(8u32).to_le_bytes());
        flags.extend_from_slice(&(MI_INT32).to_le_bytes());
        flags.extend_from_slice(&0u32.to_le_bytes());
        matrix_payload.extend_from_slice(&flags);

        let mut dims = Vec::new();
        dims.extend_from_slice(&rows.to_le_bytes());
        dims.extend_from_slice(&cols.to_le_bytes());
        append_tagged(&mut matrix_payload, MI_INT32, &dims);

        append_tagged(&mut matrix_payload, MI_INT8, name.as_bytes());

        let mut real = Vec::with_capacity(values.len() * 4);
        for val in values {
            real.extend_from_slice(&val.to_le_bytes());
        }
        append_tagged(&mut matrix_payload, MI_INT32, &real);

        let mut out = Vec::new();
        out.extend_from_slice(&MI_MATRIX.to_le_bytes());
        out.extend_from_slice(&(matrix_payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&matrix_payload);
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        out
    }

    fn write_temp_mat(arrays: &[Vec<u8>]) -> PathBuf {
        let mut file_bytes = vec![0u8; HEADER_BYTES];
        let header_text = b"MATLAB 5.0 MAT-file";
        file_bytes[..header_text.len()].copy_from_slice(header_text);
        file_bytes[124] = 0;
        file_bytes[125] = 1;
        file_bytes[126] = b'I';
        file_bytes[127] = b'M';
        for arr in arrays {
            file_bytes.extend_from_slice(arr);
        }

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nuxplore_test_{stamp}.mat"));
        let mut fh = std::fs::File::create(&path).expect("create temp mat");
        fh.write_all(&file_bytes).expect("write temp mat");
        path
    }

    #[test]
    fn test_load_instance_map_prefers_default_keys() {
        let ignore_vals = vec![9, 9, 9, 9];
        let map_vals = vec![1, 2, 3, 4];
        let arr1 = make_numeric_matrix("foo", 2, 2, &ignore_vals);
        let arr2 = make_numeric_matrix("inst_map", 2, 2, &map_vals);
        let path = write_temp_mat(&[arr1, arr2]);

        let (map, key) = load_instance_map(&path, None).expect("load map");
        assert_eq!(key, "inst_map");
        assert_eq!(map.shape(), &[2, 2]);
        assert_eq!(map[[0, 0]], 1);
        assert_eq!(map[[1, 0]], 2);
        assert_eq!(map[[0, 1]], 3);
        assert_eq!(map[[1, 1]], 4);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_instance_map_with_preferred_key() {
        let arr1 = make_numeric_matrix("inst_map", 2, 2, &[1, 1, 1, 1]);
        let arr2 = make_numeric_matrix("labels", 2, 2, &[5, 6, 7, 8]);
        let path = write_temp_mat(&[arr1, arr2]);

        let (map, key) = load_instance_map(&path, Some("labels")).expect("load key");
        assert_eq!(key, "labels");
        assert_eq!(map[[0, 0]], 5);
        assert_eq!(map[[1, 1]], 8);

        let _ = std::fs::remove_file(path);
    }
}
