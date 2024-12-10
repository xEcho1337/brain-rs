use crate::training::{DataRow, DataSet};

impl DataSet {
    pub fn new(rows: Vec<DataRow>) -> Self {
        Self { rows }
    }
}