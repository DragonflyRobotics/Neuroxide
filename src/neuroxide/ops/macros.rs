#[macro_export]
macro_rules! add {
    ($a: expr, $b: expr) => {
        neuroxide::ops::add::AddOp::forward(&vec![&$a, &$b])
    };
}

#[macro_export]
macro_rules! sub {
    ($a: expr, $b: expr) => {
        neuroxide::ops::sub::SubOp::forward(&vec![&$a, &$b])
    };
}

#[macro_export]
macro_rules! mul {
    ($a: expr, $b: expr) => {
        neuroxide::ops::mul::MulOp::forward(&vec![&$a, &$b])
    };
}

#[macro_export]
macro_rules! div {
    ($a: expr, $b: expr) => {
        neuroxide::ops::div::DivOp::forward(&vec![&$a, &$b])
    };
}

#[macro_export]
macro_rules! pow {
    ($a: expr, $b: expr) => {
        neuroxide::ops::pow::PowOp::forward(&vec![&$a, &$b])
    };
}

#[macro_export]
macro_rules! sin {
    ($a: expr) => {
        neuroxide::ops::sin::SinOp::forward(&vec![&$a])
    };
}

#[macro_export]
macro_rules! cos {
    ($a: expr) => {
        neuroxide::ops::cos::CosOp::forward(&vec![&$a])
    };
}

#[macro_export]
macro_rules! ln {
    ($a: expr) => {
        neuroxide::ops::ln::LnOp::forward(&vec![&$a])
    };
}
