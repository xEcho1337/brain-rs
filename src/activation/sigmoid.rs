use crate::activation::Activation;

#[derive(Default)]
pub struct SigmoidActivation {

}

impl Activation for SigmoidActivation {
    fn activate(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn get_derivative(&self, input: f64) -> f64 {
        self.activate(input) * (1.0 - self.activate(input))
    }
}