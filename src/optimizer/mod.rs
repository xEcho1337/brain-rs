use std::any::Any;
use crate::activation::Activation;
use crate::layer::{DenseLayer, Layer};
use crate::structure::{Neuron, Synapse};

mod adam;

const GRADIENT_CLIP: f64 = 5.0;

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    first_momentum: Vec<f64>,
    second_momentum: Vec<f64>,
    beta1_timestep: f64,
    beta2_timestep: f64,
    timestep: usize,
}

pub trait Optimizer: Any {
    fn new(learning_rate: f64) -> Self where Self: Sized;

    fn post_initialize(&mut self) {}

    fn get_learning_rate(&mut self) -> f64;

    fn set_learning_rate(&mut self, learning_rate: f64);

    fn update(&mut self, synapse: &mut Synapse);

    fn post_iteration(&mut self, layers: &[dyn Layer]);

    fn post_fit(&mut self, layers: &[dyn Layer]) {}

    fn apply_gradient_step(
        &mut self,
        layer: DenseLayer,
        neuron: &mut Neuron,
        synapse: &mut Synapse,
    ) {
        let output = neuron.value;
        let activation = &layer.activation;

        let error = self.clip_gradient(synapse.weight * synapse.output_neuron.delta);
        let delta = self.clip_gradient(error * activation.get_derivative(output));

        let weight_change = self.clip_gradient(delta * synapse.input_neuron.value);

        neuron.delta = delta;
        synapse.weight += weight_change;
    }

    fn clip_gradient(&self, gradient: f64) -> f64 {
        gradient.clamp(-5.0, 5.0)
    }
}
