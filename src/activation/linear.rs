use crate::activation::Activation;

const ONE: f64 = 1.;

pub struct LinearActivation {

}

impl Activation for LinearActivation {
    fn activate(input: f64) -> f64 {
        input
    }

    fn get_derivative(input: f64) -> f64 {
        ONE
    }

    fn activate_from_inputs(input: Vec<f64>) -> Vec<f64> {
        input
    }
}