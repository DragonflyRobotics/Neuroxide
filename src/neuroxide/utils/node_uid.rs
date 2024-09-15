use rand::Rng;


pub fn make_node_uid() -> i32 {
    return rand::thread_rng().r#gen()
}
