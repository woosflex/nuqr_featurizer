use ndarray::{Array2, Array3};
use nuxplore::io::image::load_rgb_image;
use nuxplore::save_cropped_nuclei_from_instance_map;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{stamp}"))
}

#[test]
fn test_crop_export_masking_and_layout() {
    let mut image = Array3::<u8>::zeros((6, 6, 3));
    for r in 0..6 {
        for c in 0..6 {
            image[[r, c, 0]] = (r * 10 + c) as u8;
            image[[r, c, 1]] = (r * 7 + c * 2) as u8;
            image[[r, c, 2]] = (200usize.saturating_sub(r * 5 + c)) as u8;
        }
    }

    let mut instance_map = Array2::<u32>::zeros((6, 6));
    // Label 2 intentionally inserted before label 1 to confirm sorted output by label.
    instance_map[[4, 4]] = 2;
    instance_map[[4, 5]] = 2;
    // Label 1 touches image boundary and leaves holes in padded box for masking checks.
    instance_map[[0, 0]] = 1;
    instance_map[[0, 1]] = 1;
    instance_map[[1, 0]] = 1;

    let out_dir = unique_temp_dir("nuxplore_crop_export");
    let records = save_cropped_nuclei_from_instance_map(
        &image.view(),
        &instance_map.view(),
        &out_dir,
        1,
        true,
        false,
    )
    .expect("save crops");

    assert_eq!(records.len(), 2);
    assert_eq!(records[0].nucleus_id, 1);
    assert_eq!(records[1].nucleus_id, 2);
    assert!(records[0].pre_path.is_some());
    assert!(records[0].post_path.is_none());

    let first = &records[0];
    let pre_path = first.pre_path.as_ref().expect("pre path exists");
    assert!(pre_path.exists());
    assert!(pre_path.to_string_lossy().contains("pre_normalized_nuclei"));
    assert!(pre_path.to_string_lossy().ends_with("nucleus_0001.png"));

    let patch = load_rgb_image(pre_path).expect("read saved patch");
    let (min_r, min_c, max_r, max_c) = first.bbox;
    assert_eq!(patch.dim().0, max_r - min_r + 1);
    assert_eq!(patch.dim().1, max_c - min_c + 1);

    for local_r in 0..patch.dim().0 {
        for local_c in 0..patch.dim().1 {
            let global_r = min_r + local_r;
            let global_c = min_c + local_c;
            let belongs_to_nucleus = instance_map[[global_r, global_c]] == first.nucleus_id;
            if belongs_to_nucleus {
                assert_eq!(patch[[local_r, local_c, 0]], image[[global_r, global_c, 0]]);
                assert_eq!(patch[[local_r, local_c, 1]], image[[global_r, global_c, 1]]);
                assert_eq!(patch[[local_r, local_c, 2]], image[[global_r, global_c, 2]]);
            } else {
                assert_eq!(patch[[local_r, local_c, 0]], 0);
                assert_eq!(patch[[local_r, local_c, 1]], 0);
                assert_eq!(patch[[local_r, local_c, 2]], 0);
            }
        }
    }

    let _ = std::fs::remove_dir_all(out_dir);
}
