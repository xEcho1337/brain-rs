use crate::activation::Activation;

pub struct SigmoidActivation {

}

impl Activation for SigmoidActivation {
    fn activate(input: f64) -> f64 {
        1. / (1. + (-input).exp())
    }

    fn get_derivative(input: f64) -> f64 {
        Self::activate(input) * (1. - Self::activate(input))
    }
}