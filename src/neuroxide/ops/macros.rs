#[macro_export]
macro_rules! add {
    ($a: expr, $b: expr) => {
        neuroxide::ops::add::AddOp::forward(&vec![&$a, &$b])
    };
}
