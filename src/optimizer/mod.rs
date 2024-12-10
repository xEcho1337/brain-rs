mod adam;

use crate::layer::{DenseLayer, Layer};
use crate::structure::{Neuron, Synapse};

const GRADIENT_CLIP: f64 = 5.;

pub trait Optimizer {
    fn new(learning_rate: f64) -> Self;

    fn post_initialize(&mut self) {}

    fn get_learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, learning_rate: f64);

    fn update(&mut self, synapse: &mut Synapse);

    fn post_iteration(&mut self, layers: &[DenseLayer]);

    fn post_fit(&mut self, layers: &[DenseLayer]) {}

    fn apply_gradient_step(&mut self, layer: &DenseLayer, neuron: &mut Neuron, synapse: &mut Synapse) {
        let output = neuron.get_value();

        let error = Self::clip_gradient(synapse.weight * synapse.output_neuron.delta);
        let delta = Self::clip_gradient(error * layer.activation.get_function().get_derivative(output)); // TODO

        let weight_change = Self::clip_gradient(delta * synapse.input_neuron.value);

        neuron.delta = delta;
        synapse.weight += weight_change;
    }

    fn clip_gradient(gradient: f64) -> f64 {
        gradient.clamp(-GRADIENT_CLIP, GRADIENT_CLIP)
    }
}