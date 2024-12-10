use crate::activation::Activation;

pub struct TanhActivation {

}

impl Activation for TanhActivation {
    fn activate(input: f64) -> f64 {
        input.tanh()
    }

    fn get_derivative(input: f64) -> f64 {
        1. - input.tanh().powi(2)
    }
}