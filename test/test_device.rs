use neuroxide::types::device::Device;

#[test]
fn print_device() {
    assert!(Device::CPU.to_string() == "Device::CPU");
    assert!(Device::CUDA.to_string() == "Device::CUDA");
}
