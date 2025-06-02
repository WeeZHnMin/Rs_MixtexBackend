pub fn check_repetition<T: PartialEq + Copy>(arr: &[T], repeats: usize) -> bool {
    let length = arr.len();
    if length < repeats {
        return false;
    }

    // 枚举所有可能的 pattern_length
    for pattern_length in 1..=(length / repeats) {
        let total_len = pattern_length * repeats;
        if total_len > length {
            break;
        }
        // 取末尾 repeats*pattern_length 个元素
        let repeated_section = &arr[length - total_len..];

        // 检查每段是否都等于第一段
        let first_pattern = &repeated_section[..pattern_length];
        let mut all_equal = true;
        for i in 1..repeats {
            let start = i * pattern_length;
            let end = start + pattern_length;
            if repeated_section[start..end] != *first_pattern {
                all_equal = false;
                break;
            }
        }
        if all_equal {
            return true;
        }
    }
    false
}