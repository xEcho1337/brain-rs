use crate::layer::Layer;
use crate::model::Model;

impl Model {
    pub fn new(layers: Vec<Layer>) {
        Self { layers };
    }
}