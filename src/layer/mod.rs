use std::any::Any;

use crate::activation::{Activation};
use crate::structure::{Neuron, Synapse};

mod dense_layer;
mod dropout;
mod layer_norm;

pub trait Layer: Any {
    //fn get_neurons(&self) -> &mut Vec<Neuron>;

    fn as_any(&self) -> &dyn Any;

    fn apply_function(&self, previous: dyn Layer);
}

pub struct DenseLayer {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub activation: dyn Activation
}