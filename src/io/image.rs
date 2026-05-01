use crate::core::{FeaturizerError, Result};
use ndarray::{Array3, ArrayView3};
use std::path::Path;

pub fn load_rgb_image(path: &Path) -> Result<Array3<u8>> {
    let dyn_img = image::open(path).map_err(|err| {
        FeaturizerError::InvalidInput(format!("failed to open image '{}': {err}", path.display()))
    })?;
    let rgb = dyn_img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let data = rgb.into_raw();
    Array3::from_shape_vec((height as usize, width as usize, 3), data).map_err(|err| {
        FeaturizerError::InvalidInput(format!(
            "failed to reshape RGB image '{}' into array: {err}",
            path.display()
        ))
    })
}

pub fn save_rgb_image(path: &Path, rgb: &ArrayView3<'_, u8>) -> Result<()> {
    let (height, width, channels) = rgb.dim();
    if channels != 3 {
        return Err(FeaturizerError::InvalidInput(format!(
            "expected RGB array with 3 channels, got {channels}"
        )));
    }
    if width == 0 || height == 0 {
        return Err(FeaturizerError::InvalidInput(
            "cannot save empty RGB image".to_string(),
        ));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (raw, offset) = rgb.to_owned().into_raw_vec_and_offset();
    let data = if offset.is_some() {
        rgb.iter().copied().collect()
    } else {
        raw
    };
    let img = image::RgbImage::from_raw(width as u32, height as u32, data).ok_or_else(|| {
        FeaturizerError::InvalidInput(format!(
            "failed to construct RGB image buffer for '{}'",
            path.display()
        ))
    })?;
    img.save(path).map_err(|err| {
        FeaturizerError::InvalidInput(format!("failed to save image '{}': {err}", path.display()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_load_rgb_image_roundtrip_png() {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(2, 2);
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.put_pixel(0, 1, Rgb([70, 80, 90]));
        img.put_pixel(1, 1, Rgb([100, 110, 120]));

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nuxplore_test_{stamp}.png"));
        img.save(&path).expect("write temp png");

        let loaded = load_rgb_image(&path).expect("load temp png");
        assert_eq!(loaded.shape(), &[2, 2, 3]);
        assert_eq!(loaded[[0, 0, 0]], 10);
        assert_eq!(loaded[[0, 1, 2]], 60);
        assert_eq!(loaded[[1, 0, 1]], 80);
        assert_eq!(loaded[[1, 1, 0]], 100);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_rgb_image_roundtrip_png() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nuxplore_test_save_{stamp}.png"));

        let mut arr = Array3::<u8>::zeros((2, 2, 3));
        arr[[0, 0, 0]] = 1;
        arr[[0, 0, 1]] = 2;
        arr[[0, 0, 2]] = 3;
        arr[[1, 1, 0]] = 250;
        arr[[1, 1, 1]] = 251;
        arr[[1, 1, 2]] = 252;

        save_rgb_image(&path, &arr.view()).expect("save temp png");
        let loaded = load_rgb_image(&path).expect("load saved temp png");
        assert_eq!(loaded.shape(), &[2, 2, 3]);
        assert_eq!(loaded[[0, 0, 0]], 1);
        assert_eq!(loaded[[0, 0, 1]], 2);
        assert_eq!(loaded[[0, 0, 2]], 3);
        assert_eq!(loaded[[1, 1, 0]], 250);
        assert_eq!(loaded[[1, 1, 1]], 251);
        assert_eq!(loaded[[1, 1, 2]], 252);

        let _ = std::fs::remove_file(path);
    }
}
