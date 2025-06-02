use image::DynamicImage;

pub struct TemporaryData {
    pub image: DynamicImage,
    pub token_id_array: Vec<u32>,
}

#[allow(dead_code)]
#[allow(unused_imports)]
impl TemporaryData {
    /// Creates a new TemporaryData instance.
    pub fn new() -> Self {
        // 创建一张 1x1 的透明图片
        let image = DynamicImage::new_rgba8(1, 1);
        let token_id_array = Vec::new();
        Self { image, token_id_array }
    }

    /// Returns a reference to the image.
    pub fn get_image(&self) -> Option<&DynamicImage> {
        Some(&self.image)
    }

    pub fn set_image(&mut self, image: DynamicImage) {
        self.image = image;
    }

    pub fn set_token_id_array(&mut self, token_id_array: Vec<u32>) {
        self.token_id_array = token_id_array;
    }

    /// Returns a reference to the token_id_array.
    pub fn token_id_array(&self) -> &Vec<u32> {
        &self.token_id_array
    }

    /// Adds a token id to the array.
    pub fn add_token_id(&mut self, token_id: u32) {
        self.token_id_array.push(token_id);
    }

    /// Removes all token ids.
    pub fn clear_token_ids(&mut self) {
        self.token_id_array.clear();
    }

    /// Returns the number of token ids.
    pub fn token_count(&self) -> usize {
        self.token_id_array.len()
    }
}