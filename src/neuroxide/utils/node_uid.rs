use rand::Rng;


pub fn makeNodeUID() -> i32 {
    return rand::thread_rng().r#gen()
}
