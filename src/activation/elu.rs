use crate::activation::Activation;

pub struct EluActivation {
    alpha: f64,
}

impl EluActivation {
    fn new(&self, alpha: f64) -> self {
        self { alpha }
    }
}

impl Activation for EluActivation {
    fn activate(input: f64) -> f64 {
        if input > 0. {
            input
        } else {
            Self.alpha * (input.exp() - 1.)
        }
    }

    fn get_derivative(input: f64) -> f64 {
        todo!()
    }
}