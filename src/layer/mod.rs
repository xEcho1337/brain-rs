use crate::activation::Activations;
use crate::structure::{Neuron, Synapse};

mod dense_layer;
mod dropout;
mod layer_norm;

pub trait Layer {
    fn apply_function(previous: dyn Layer);
}

pub struct DenseLayer {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub activation: Activations
}