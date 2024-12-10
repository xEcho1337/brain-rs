use crate::activation::Activation;

pub struct SigmoidActivation {

}

impl Activation for SigmoidActivation {
    fn activate(input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn get_derivative(input: f64) -> f64 {
        Self::activate(input) * (1.0 - Self::activate(input))
    }
}