use rayon::iter::IntoParallelRefIterator;

use crate::structure::Neuron;

pub mod elu;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

/*
pub enum Activations {
    EluActivation(EluActivation),
    Linear(LinearActivation),
    ReLU(ReLUActivation),
    Sigmoid(SigmoidActivation),
    Softmax(SoftmaxActivation),
    Tanh(TanhActivation),
}
*/

pub trait Activation {
    fn activate(&self, input: f64) -> f64 where Self: Sized;

    fn get_derivative(&self, input: f64) -> f64 where Self: Sized;

    fn activate_from_inputs(&self, input: Vec<f64>) -> Vec<f64> {
        input
            .par_iter()
            .map(|value| {
                self.activate(value);
            })
            .collect()
    }

    fn apply(&self, neurons: Vec<Neuron>) {
        let _ = neurons.par_iter().for_each(|neuron: &mut Neuron| {
            let output = self.activate(neuron.value + neuron.bias);
            neuron.value = output;
        });
    }
}
