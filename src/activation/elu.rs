use crate::activation::Activation;

#[derive(Default)]
pub struct EluActivation {
    alpha: f64,
}

impl EluActivation {
    fn new(&self, alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Activation for EluActivation {
    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 {
            input
        } else {
            self.alpha * (input.exp() - 1.0)
        }
    }

    fn get_derivative(&self, input: f64) -> f64 {
        todo!()
    }
}
