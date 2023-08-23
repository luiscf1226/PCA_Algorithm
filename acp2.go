package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

const (
	matrixSize = 5
	precision  = 1e-6
	maxIter    = 1000
)

func main() {
	filePath := "EjemploEstudiantes.csv"
	matrix, err := readCSVFile(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Matrix Received by file:")
	printMatrix(matrix)

	means := calculateColumnMeans(matrix)
	stdDevs := calculateColumnStdDevs(matrix)
	normalizedMatrix := normalizeMatrix(matrix, means, stdDevs)
	fmt.Println("Matrix Normalized:")
	printMatrix(normalizedMatrix)

	correlationMatrix := calculateCorrelationMatrix(normalizedMatrix)
	fmt.Println("Correlation Matrix:")
	printMatrix(correlationMatrix)

	values, vectors := calculateEigenvaluesAndEigenvectors(correlationMatrix)
	fmt.Println("Eigenvalues:", values)
	fmt.Println("Eigenvectors:")
	for _, vector := range vectors {
		fmt.Println(vector)
	}

	eigenMatrix := calculateEigenMatrix(vectors)
	fmt.Println("Eigen matrix:")
	printMatrix(eigenMatrix)

	pcMatrix := calculatePCMatrix(normalizedMatrix, eigenMatrix)
	fmt.Println("Principal Component matrix:")
	printMatrix(pcMatrix)

	individualmatrix := calculateIndividualMatrix(pcMatrix, normalizedMatrix)
	fmt.Println("Individual Qualities Matrix:")
	printMatrix(individualmatrix)

	coordinateValues, coordinatesMatrix := calculateAutoValuesandVectors(pcMatrix)
	fmt.Println("Coordinates Values from Eigen values:")
	for _, vector := range coordinateValues {
		fmt.Println(vector)
	}
	fmt.Println("Coordinates Matrix:")
	printMatrix(coordinatesMatrix)
}

func readCSVFile(filePath string) ([][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matrix [][]float64
	scanner := bufio.NewScanner(file)

	if scanner.Scan() {
		scanner.Text()
	}

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, ";")

		var row []float64
		for _, field := range fields[1:] {
			value, err := strconv.ParseFloat(strings.ReplaceAll(field, ",", "."), 64)
			if err != nil {
				return nil, err
			}
			row = append(row, value)
		}
		matrix = append(matrix, row)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return matrix, nil
}

func calculateColumnMeans(matrix [][]float64) []float64 {
	means := make([]float64, len(matrix[0]))
	for _, row := range matrix {
		for i, val := range row {
			means[i] += val
		}
	}
	for i := range means {
		means[i] /= float64(len(matrix))
	}
	return means
}

func calculateColumnStdDevs(matrix [][]float64) []float64 {
	means := calculateColumnMeans(matrix)
	stdDevs := make([]float64, len(matrix[0]))
	for _, row := range matrix {
		for i, val := range row {
			stdDevs[i] += math.Pow(val-means[i], 2)
		}
	}
	for i := range stdDevs {
		stdDevs[i] = math.Sqrt(stdDevs[i] / float64(len(matrix)))
	}
	return stdDevs
}

func normalizeMatrix(matrix [][]float64, means, stdDevs []float64) [][]float64 {
	for i, row := range matrix {
		for j, val := range row {
			matrix[i][j] = (val - means[j]) / stdDevs[j]
		}
	}
	return matrix
}

func printMatrix(matrix [][]float64) {
	for _, row := range matrix {
		fmt.Println(row)
	}
}

func calculateCorrelationMatrix(normalizedMatrix [][]float64) [][]float64 {
	numCols := len(normalizedMatrix[0])
	correlationMatrix := mat.NewDense(numCols, numCols, nil)
	for i := 0; i < numCols; i++ {
		for j := 0; j < numCols; j++ {
			correlation := calculateCorrelationValue(normalizedMatrix[i], normalizedMatrix[j])
			correlationMatrix.Set(i, j, correlation)
		}
	}
	return mat.DenseCopyOf(correlationMatrix).RawMatrix().Data
}

func calculateCorrelationValue(col1, col2 []float64) float64 {
	n := len(col1)
	numerator := 0.0
	denominator1 := 0.0
	denominator2 := 0.0

	for i := 0; i < n; i++ {
		numerator += (col1[i]) * (col2[i])
		denominator1 += math.Pow(col1[i], 2)
		denominator2 += math.Pow(col2[i], 2)
	}
	correlation := numerator / (math.Sqrt(denominator1) * math.Sqrt(denominator2))
	return correlation
}

func calculateEigenvaluesAndEigenvectors(R [][]float64) ([]float64, [][]float64) {
	rows := len(R)
	values := make([]float64, rows)
	vectors := make([][]float64, rows)

	Rmat := mat.NewDense(rows, rows, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			Rmat.Set(i, j, R[i][j])
		}
	}

	var eig mat.Eigen
	eig.Factorize(Rmat, mat.EigenRight)
	eigenvalues := eig.Values(nil)
	eigenvectors := eig.VectorsTo(nil)

	for i := 0; i < rows; i++ {
		values[i] = real(eigenvalues[i])
		vectors[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			vectors[i][j] = eigenvectors.At(j, i)
		}
	}

	return values, vectors
}

func normalizeVector(vector []float64) []float64 {
	//get magnitude from function dot product squared
	magnitude := math.Sqrt(dotProduct(vector, vector))
	//each vector needs to be divided by magnitud to be normalized
	for i := range vector {
		vector[i] /= magnitude
	}
	return vector
}

func multiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	rows, cols := len(matrix), len(matrix[0])
	result := make([]float64, rows)
	// return vector by multiplying by matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i] += matrix[i][j] * vector[j]
		}
	}

	return result
}

// dot product function
func dotProduct(vector1, vector2 []float64) float64 {
	//obtains vectors and multiplies them
	result := 0.0
	for i := range vector1 {
		result += vector1[i] * vector2[i]
	}
	return result
}

func calculateEigenMatrix(eigenVectors [][]float64) [][]float64 {
	numCols := len(eigenVectors)
	eigenMatrix := make([][]float64, numCols)
	//transpose matrix change columns for rows
	for i := 0; i < numCols; i++ {
		//need to reserve space of rows of number of columns each
		eigenMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			eigenMatrix[i][j] = eigenVectors[j][i]
		}
	}
	return eigenMatrix
}

func calculatePCMatrix(normalizedMatrix [][]float64, eigenMatrix [][]float64) [][]float64 {
	//get number of rows and columnds
	numRows := len(normalizedMatrix)
	numCols := len(eigenMatrix[0])
	pcMatrix := make([][]float64, numRows)

	//set value to matrix from dot product with normalized and eigen matrix
	for i := 0; i < numRows; i++ {
		pcMatrix[i] = make([]float64, numCols)
		rowX := rowToArray(normalizedMatrix, i)

		for j := 0; j < numCols; j++ {
			columnV := columnToArray(eigenMatrix, j)
			valuedotProduct := dotProduct(rowX, columnV)
			pcMatrix[i][j] = valuedotProduct
		}
	}

	return pcMatrix
}
func calculateIndividualMatrix(principalMatrix [][]float64, normalizedMatrix [][]float64) [][]float64 {
	numRows1 := len(normalizedMatrix)
	numCols1 := len(normalizedMatrix[0])
	denominator := 0.0
	matrixXvalue := 0.0
	individualMatrix := make([][]float64, numRows1)

	for i := 0; i < numRows1; i++ {
		for j := 0; j < numCols1; j++ {
			matrixXvalue = math.Pow(normalizedMatrix[i][j], 2)
			denominator += matrixXvalue
		}
	}

	totalvalue := 0.0
	matrixCvalue := 0.0

	for i := 0; i < numRows1; i++ {
		individualMatrix[i] = make([]float64, numCols1)
		for j := 0; j < numCols1; j++ {
			matrixCvalue = math.Pow(principalMatrix[i][j], 2)
			totalvalue = matrixCvalue / denominator
			individualMatrix[i][j] = totalvalue
		}
	}
	return individualMatrix
}

func columnToArray(matrix [][]float64, j int) []float64 {
	//return column from matrix given a index
	numRows := len(matrix)
	array := make([]float64, numRows)
	for i := 0; i < numRows; i++ {
		array[i] = matrix[i][j]
	}
	return array
}

func rowToArray(matrix [][]float64, i int) []float64 {
	// function to return row as an array given an index
	numCols := len(matrix[i])
	array := make([]float64, numCols)
	copy(array, matrix[i])
	return array
}

func calculateAutoValuesandVectors(pcMatrix [][]float64) ([]float64, [][]float64) {
	correlationPCmatrix := calculateCorrelationMatrix(pcMatrix)
	//obtain list of proper values and proper vectors
	values, vectors := calculateEigenvaluesAndEigenvectors(correlationPCmatrix)
	eigenMatrix := calculateEigenMatrix(vectors)
	coordinatesMatrix := calculateCoordinatesMatrix(pcMatrix, eigenMatrix)
	return values, coordinatesMatrix

}
func calculateCoordinatesMatrix(pcMatrix, eigenMatrix [][]float64) [][]float64 {
	numRows := len(pcMatrix)
	numCols := len(eigenMatrix[0])
	coordinatesMatrix := make([][]float64, numRows)
	//receive pc an eigen matrix to get dor product of each
	for i := 0; i < numRows; i++ {
		coordinatesMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			//insert dot product value in coordinates matrix
			coordinatesMatrix[i][j] = dotProduct(pcMatrix[i], eigenMatrix[j])
		}
	}

	return coordinatesMatrix
}
