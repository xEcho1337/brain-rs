use crate::activation::Activation;

const ONE: f64 = 1.;

#[derive(Default)]
pub struct LinearActivation {

}

impl Activation for LinearActivation {
    fn activate(&self, input: f64) -> f64 {
        input
    }

    fn get_derivative(&self, input: f64) -> f64 {
        ONE
    }

    fn activate_from_inputs(&self, input: Vec<f64>) -> Vec<f64> {
        input
    }
}