use crate::activation::Activation;

pub struct ReLUActivation {
}

impl Activation for ReLUActivation {
    fn activate(input: f64) -> f64 {
        input.max(0.)
    }

    fn get_derivative(input: f64) -> f64 {
        if input > 0. { 1. } else { 0. }
    }
}