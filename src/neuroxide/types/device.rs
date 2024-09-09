#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::CPU => write!(f, "Device::CPU"),
            Self::CUDA => write!(f, "Device::CUDA")
        }
    }
}
    
