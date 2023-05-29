pub fn matrix_transpose (matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transpose_matrix: Vec<Vec<f64>> = Vec::new();
    for _row in 0..matrix[0].len(){
        let mut vector: Vec<f64> = Vec::new();
        for _column in 0..matrix.len(){
            vector.push(0.0);
        }
        transpose_matrix.push(vector);
    }
    for row in 0..matrix.len(){
        for column in 0..matrix[0].len(){
            transpose_matrix[column][row] = matrix[row][column];
        }
    }
    transpose_matrix
}