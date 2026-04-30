use crate::core::{FeaturizerError, Result};
use ndarray::Array3;
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
}
