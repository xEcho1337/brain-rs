use crate::utils::Vector;

mod datarow;
mod dataset;

pub struct DataRow {
    input: Vector,
    output: Vector,
}

pub struct DataSet {
    rows: Vec<DataRow>,
}
