use crate::structure::Neuron;
use rayon::iter::IntoParallelRefIterator;

mod relu;
mod linear;
mod sigmoid;
mod softmax;
mod tanh;
mod elu;

pub enum Activations {
    Linear,
    Elu,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
}

pub trait Activation {
    fn activate(input: f64) -> f64;

    fn get_derivative(input: f64) -> f64;

    fn activate_from_inputs(input: Vec<f64>) -> Vec<f64> {
        input.par_iter()
            .map(|value| {
                Self::activate(value);
            })
            .collect();
    }

    fn apply(neurons: Vec<Neuron>) {
        let _ = neurons
            .par_iter()
            .for_each(|neuron: &mut Neuron| {
                let output = Self::activate(neuron.value + neuron.bias);
                neuron.value = output;
            });
    }
}