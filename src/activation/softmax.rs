use crate::activation::Activation;
use rayon::iter::IntoParallelRefIterator;

pub struct SoftmaxActivation {

}

impl Activation for SoftmaxActivation {
    fn activate(input: f64) -> f64 {
        panic!("activate(input: f64) is not supported for Softmax.")
    }

    fn get_derivative(input: f64) -> f64 {
        input * (1.0 - input)
    }

    fn activate_from_inputs(input: Vec<f64>) -> Vec<f64> {
        let max_input = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut exp_values = Vec::with_capacity(input.len());
        let mut sum = 0.0;

        input.iter().for_each(|&neuron| {
            let exp_val = (neuron - max_input).exp();

            exp_values.push(exp_val);

            sum += exp_val;
        });

        exp_values.iter_mut().for_each(|val| *val /= sum);

        exp_values
    }

    fn apply(mut neurons: Vec<Neuron>) {
        let values: Vec<f64> = Vec::with_capacity(neurons.len());

        neurons
            .par_iter()
            .map(|neuron| neuron.value + neuron.bias)
            .collect_into_vec(&values);

        let activated_values = Self::activate_from_inputs(values);

        neurons
            .par_iter()
            .enumerate()
            .for_each(|(i, neuron)| {
                neuron.value = activated_values[i];
            });
    }
}