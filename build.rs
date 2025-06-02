fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("icon.ico"); // 替换成你的图标路径
    res.compile().unwrap();
}
