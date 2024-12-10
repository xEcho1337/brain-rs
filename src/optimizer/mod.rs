use std::any::Any;

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

pub trait Optimizer {
    fn new(&self, learning_rate: f64) -> Self;

    fn post_initialize(&mut self) {}

    fn get_learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, learning_rate: f64);

    fn update(&mut self, synapse: &mut Synapse);

    fn post_iteration(&mut self, layers: &[DenseLayer]);

    fn post_fit(&mut self, layers: &[DenseLayer]) {}

    fn apply_gradient_step(
        self,
        layer: &DenseLayer,
        neuron: &mut Neuron,
        synapse: &mut Synapse,
    ) {
        let output = neuron.value;

        let error = self.clip_gradient(synapse.weight * synapse.output_neuron.delta);
        let delta = self.clip_gradient(error * layer.activation.get_derivative(output));

        let weight_change = self.clip_gradient(delta * synapse.input_neuron.value);

        neuron.delta = delta;
        synapse.weight += weight_change;
    }

    fn clip_gradient(&self, gradient: f64) -> f64 {
        gradient.clamp(-GRADIENT_CLIP, GRADIENT_CLIP)
    }
}
