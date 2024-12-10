use crate::initialization::WeightInitialization;
use crate::layer::Layer;
use crate::loss::LossFunctions;

mod model;

pub struct Model {
    layers: Vec<dyn Layer>,
    init_technique: Option<WeightInitialization>,
    loss_functions: Option<LossFunctions>
}