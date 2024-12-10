use crate::activation::Activation;

#[derive(Default)]
pub struct TanhActivation {

}

impl Activation for TanhActivation {
    fn activate(&self, input: f64) -> f64 {
        input.tanh()
    }

    fn get_derivative(&self, input: f64) -> f64 {
        1.0 - input.tanh().powi(2)
    }
}