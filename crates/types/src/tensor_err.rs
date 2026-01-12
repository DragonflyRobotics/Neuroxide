#[derive(Debug)]
pub enum TensorCreationError {
    InvalidTensorDataCPU,
}

impl std::fmt::Display for TensorCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorCreationError::InvalidTensorDataCPU => {
                write!(f, "Invalid tensor data or device for CPU tensor creation!")
            }
        }
    }
}

impl std::error::Error for TensorCreationError {}
