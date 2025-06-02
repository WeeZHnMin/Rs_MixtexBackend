//src//process_img.rs
use image::{
    imageops::FilterType,
    Rgba,
    ImageBuffer,
    RgbaImage,
    DynamicImage, // Retain DynamicImage import as it's the return type of decode()
    GenericImageView, // <-- FIX 2: Import GenericImageView trait
};

use image::buffer::ConvertBuffer; // <-- FIX 3: Import ConvertBuffer trait
use ndarray::{Array3, Axis, ArrayBase, OwnedRepr, Dim};
use anyhow::Context;

pub fn process_image_with_padding(img: DynamicImage, _save_path: &str) -> anyhow::Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
    // 2. Resize the image, keeping the aspect ratio
    // Use dimensions() method from GenericImageView trait
    let (orig_width, orig_height) = img.dimensions(); // This call now works after importing GenericImageView
    let (new_width, new_height) = if orig_width > orig_height {
        (448, (orig_height as f32 * 448.0 / orig_width as f32).round() as u32)
    } else {
        ((orig_width as f32 * 448.0 / orig_height as f32).round() as u32, 448)
    };

    // Resize the original dynamic image
    let resized = img.resize_exact(new_width, new_height, FilterType::CatmullRom); // resized is DynamicImage

    // Convert the resized image to RGBA for overlay compatibility
    let resized_rgba = resized.to_rgba8(); // resized_rgba is RgbaImage


    // 3. Create a new image with a white background (448x448), using RGBA
    let mut padded_img_rgba = RgbaImage::new(448, 448);
    for (_x, _y, pixel) in padded_img_rgba.enumerate_pixels_mut() {
        *pixel = Rgba([255, 255, 255, 255]); // White background
    }

    // 4. Calculate the offset to center the resized image
    let offset_x = (448 - new_width) / 2;
    let offset_y = (448 - new_height) / 2;

    // 5. Place the resized RGBA image onto the white RGBA background
    // Use u32 offsets converted to i64
    image::imageops::overlay(&mut padded_img_rgba, &resized_rgba, offset_x.into(), offset_y.into());

    // 6. Convert the padded image from RGBA (4 channels) to RGB (3 channels)
    // Use the convert method from ConvertBuffer trait on the instance
    let padded_img_rgb: ImageBuffer<image::Rgb<u8>, Vec<u8>> = padded_img_rgba.convert(); // <-- FIX 3: Call convert() on the instance
    let (width, height) = padded_img_rgb.dimensions(); // Use the RGB image dimensions

    // 7. Convert to ndarray (H, W, C)
    let raw_pixels = padded_img_rgb.into_raw(); // Vec<u8>, now contains H*W*3 bytes

    // 8. Convert to f32 and rescale [0, 255] -> [0, 1]
    let raw_pixels_f32: Vec<f32> = raw_pixels.iter()
        .map(|&x| x as f32 / 255.0)
        .collect();

    // 9. Create ndarray (H, W, C), shape is (height, width, 3)
    let img_array = Array3::from_shape_vec(
        (height as usize, width as usize, 3),
        raw_pixels_f32
    ).context("Failed to create ndarray from pixel data")?;

    // 10. Normalize (H, W, C)
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];

    // Create ndarrays for mean and std with shape (1, 1, 3) for broadcasting
    let mean_array = Array3::from_shape_fn((1, 1, 3), |(_, _, c)| mean[c]);
    let std_array = Array3::from_shape_fn((1, 1, 3), |(_, _, c)| std[c]);

    // Perform normalization using element-wise operations with broadcasting
    let normalized = (img_array - mean_array) / std_array;


    // 11. Rearrange dimensions to (C, H, W)
    let mut channels_first: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> = Array3::<f32>::zeros((3, height as usize, width as usize));
    for h in 0..height as usize {
        for w in 0..width as usize {
            for c in 0..3 {
                channels_first[[c, h, w]] = normalized[[h, w, c]];
            }
        }
    }

    _save_image_from_array(&channels_first, _save_path)?;

    // 11.5 Add batch dimension: (1, C, H, W)
    let batched: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = channels_first.insert_axis(Axis(0));

    Ok(batched)
}

fn _save_image_from_array(array: &Array3<f32>, file_path: &str) -> anyhow::Result<()> {
    // Input array is (C, H, W) normalized

    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];

    let (c, h, w) = (array.dim().0, array.dim().1, array.dim().2);
    if c != 3 {
        return Err(anyhow::anyhow!("Input array to save_image_from_array must have 3 channels"));
    }

    // 1. Denormalize the image and convert to HWC (RGB) Vec<u8>
    let mut denormalized_img_bytes: Vec<u8> = Vec::with_capacity(h * w * 3);

    for h in 0..h {
        for w in 0..w {
            for c in 0..c {
                // Denormalize: value = (normalized_value * std) + mean
                let pixel_f32 = array[[c, h, w]] * std[c] + mean[c];
                // Scale back to [0, 255] and clamp, then convert to u8
                let pixel_u8 = (pixel_f32 * 255.0).clamp(0.0, 255.0) as u8;
                denormalized_img_bytes.push(pixel_u8);
            }
        }
    }

    // 2. Convert the denormalized HWC (RGB) data into an RgbImage
    let img = ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w as u32, h as u32, denormalized_img_bytes)
        .context("Failed to create RgbImage from raw bytes")?;

    // 3. Save the image to a file
    img.save(file_path).context(format!("Failed to save image to {}", file_path))?;

    Ok(())
}