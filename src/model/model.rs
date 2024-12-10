use std::ptr::null;

use crate::initialization::WeightInitialization;
use crate::layer::{DenseLayer, Layer};
use crate::loss::LossFunctions;
use crate::model::Model;
use crate::optimizer::Optimizer;
use crate::utils::Vector;

impl Model {
    pub fn new(layers: Vec<impl Layer>) -> Self {
        Self {
            layers,
            init_technique: None,
            loss_functions: None,
            optimizer: None,
        }
    }

    pub fn compile(&mut self, init_technique: WeightInitialization, loss_functions: LossFunctions, optimizer: impl Optimizer) {
        self.init_technique = Some(init_technique);
        self.loss_functions = Some(loss_functions);
        self.optimizer = Some(optimizer);
    }

    pub fn predict(self, input: Vector) -> Vector {
        if self.layers == null() || self.layers.len() == 0 {
            panic!("Model has not been compiled yet.");
        }

        let input_layer = &self.layers[0];

        if let Some(dense_layer) = input_layer.as_any().downcast_ref::<DenseLayer>() {
            if dense_layer.neurons.len() != input.size() {
                panic!("Input size does not match the input dimension.");
            }

            for layer in self.layers.iter() {
                for mut neuron in layer.neurons {
                    neuron.value = 0.0;
                }
            }
        } else {
            panic!("First layer is not a dense layer.");
        }
    }
}