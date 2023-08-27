package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	maxIter    = 1000
	precision  = 1e-10 // value of precision for convergence
	matrixSize = 5
)

func main() {

	filePath := "EjemploEstudiantes.csv"
	//from file path read file or return error
	matrix, err := readCSVFile(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	print("0. Matrix Received by file \n")
	printMatrix(matrix)
	//Implement first step 1 center and reduce normalized matrix
	//obtain matrix by calculating mean and deviation standard to normalize
	normalizedMatrix := normalizeMatrix(matrix)
	print("1. Matrix Normalized \n")
	printMatrix(normalizedMatrix)

	//Second step 2 Correlation Matrix
	correlationMatrix := calculateCorrelationMatrix(normalizedMatrix)
	print("2. Correlation Matrix \n")
	printMatrix(correlationMatrix)

	//third step order by greatest to least vectors
	//obtain list of proper values and proper vectors
	values, vectors := calculateEigenvaluesAndEigenvectors(correlationMatrix)

	//print eigen and vector values
	fmt.Println("3. Eigenvalues:", values)
	fmt.Println("4. Eigenvectors:")
	vectorsTranspose := transposeMatrix(vectors)
	printMatrix(vectorsTranspose)
	//fith step make PC matrix from nornmalized matrix, and eigen matrix
	pcMatrix := calculatePCMatrix(normalizedMatrix, vectorsTranspose)
	fmt.Println("5. Principal Component matrix:")
	printMatrix(pcMatrix)
	//six stepS
	individualmatrix := calculateIndividualMatrix(pcMatrix, normalizedMatrix)
	fmt.Println("6. Individual Qualities Matrix:")
	printMatrix(individualmatrix)
	//seventh step make the correlation matrix
	coordinateValues, coordinatesMatrix := calculateEigenvaluesAndVectors(pcMatrix)
	fmt.Println("Coordinates Values from Eigen values:")
	for _, vector := range coordinateValues {
		fmt.Println(vector)
	}
	fmt.Println("7. Coordinates Matrix:")
	printMatrix(coordinatesMatrix)
	//eight step quality matrix
	qualityMatrix := calculateQualityV(vectors, coordinatesMatrix)
	fmt.Println("8. Quality Matrix: ")
	printMatrix(qualityMatrix)
	//nineth step inercia vector
	inerciaV := calculateInertiaVector(values)
	fmt.Println("9. Inercia Vector: ")
	for _, vector := range inerciaV {
		fmt.Println(vector)
	}
}

// read csv file or return error if occured
func readCSVFile(filePath string) ([][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matrix [][]float64
	scanner := bufio.NewScanner(file)

	// Ignore header because non numerical value
	if scanner.Scan() {
		scanner.Text()
	}

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, ";")

		var row []float64
		for _, field := range fields[1:] { // Omitir el primer campo nombres
			//remplazar coma por punto y darle el valor de float
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

	// Calculate the sum of each column
	for _, row := range matrix {
		for i, val := range row {
			means[i] += val
		}
	}

	// Divide by the number of rows to get the average of each column
	for i := range means {
		means[i] /= float64(len(matrix))
	}

	return means
}

func calculateColumnStdDevs(matrix [][]float64) []float64 {
	if len(matrix) == 0 {
		return nil
	}

	means := calculateColumnMeans(matrix)
	stdDevs := make([]float64, len(matrix[0]))

	for _, row := range matrix {
		for i, val := range row {
			stdDevs[i] += math.Pow(val-means[i], 2)
		}
	}

	for i := range stdDevs {
		// standard deviation formula for each value
		stdDevs[i] = math.Sqrt(stdDevs[i] / float64(len(matrix)-1)) // Remember the note about n-1 vs n
	}

	return stdDevs
}

// Normalize the matrix using z-score normalization
func normalizeMatrix(matrix [][]float64) [][]float64 {
	if len(matrix) == 0 {
		return nil
	}

	means := calculateColumnMeans(matrix)
	stdDevs := calculateColumnStdDevs(matrix)

	normalizedMatrix := make([][]float64, len(matrix))
	for i := range matrix {
		normalizedMatrix[i] = make([]float64, len(matrix[i]))
		// Subtract mean and divide by standard deviation for each value
		for j, val := range matrix[i] {
			normalizedMatrix[i][j] = (val - means[j]) / stdDevs[j]
		}
	}
	return normalizedMatrix
}

// print array
func printArray(arr []float64) {
	for _, val := range arr {
		fmt.Printf("%f ", val)
	}
	fmt.Println() // adds a new line at the end
}

// Print Matrix function
func printMatrix(matrix [][]float64) {
	for _, row := range matrix {
		fmt.Println(row)
	}
}

func calculateCorrelationMatrix(normalizedMatrix [][]float64) [][]float64 {
	numCols := len(normalizedMatrix[0])
	correlationMatrix := make([][]float64, numCols)
	for i := 0; i < numCols; i++ {
		correlationMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			// Obtain correlation value for matrix
			correlation := calculateCorrelationValue(getColumn(normalizedMatrix, i),
				getColumn(normalizedMatrix, j))
			correlationMatrix[i][j] = correlation
		}
	}
	return correlationMatrix
}

func getColumn(matrix [][]float64, colIndex int) []float64 {
	column := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		column[i] = matrix[i][colIndex]
	}
	return column
}

// simplify formula
func calculateCorrelationValue(col1, col2 []float64) float64 {
	n := len(col1)
	numerator := 0.0
	denominator1 := 0.0
	denominator2 := 0.0
	// Get both denominators from both columns
	for i := 0; i < n; i++ {
		numerator += col1[i] * col2[i]
		denominator1 += col1[i] * col1[i]
		denominator2 += col2[i] * col2[i]
	}

	// Division can't be equal to 0
	if denominator1 == 0 || denominator2 == 0 {
		return 0
	}
	// Use correlation formula
	correlation := numerator / (math.Sqrt(denominator1) * math.Sqrt(denominator2))
	return correlation
}
func calculateEigenvaluesAndEigenvectors(R [][]float64) ([]float64, [][]float64) {
	//needa to receive matrix and return values in array and vectors in matrix
	rows := len(R)
	values := make([]float64, rows)
	vectors := make([][]float64, rows)
	//get vector randomly
	for k := 0; k < rows; k++ {
		vector := make([]float64, rows)
		for j := 0; j < rows; j++ {
			vector[j] = rand.Float64()
		}
		vector = normalizeVector(vector)

		for iter := 0; iter < maxIter; iter++ {
			nextVector := multiplyMatrixVector(R, vector)
			nextVector = normalizeVector(nextVector)

			// Check for convergence of eigenvector
			if vectorDiff(nextVector, vector) < precision {
				break
			}
			vector = nextVector
		}

		values[k] = dotProduct(multiplyMatrixVector(R, vector), vector)
		vectors[k] = vector

		//deflate matrix
		//subtract product of the eigen vector
		for i := 0; i < rows; i++ {
			for j := 0; j < rows; j++ {
				//makes sure next largest eigen value and vector are paired
				R[i][j] -= values[k] * vectors[k][i] * vectors[k][j]
			}
		}
	}

	// Sort eigenvalues and corresponding eigenvectors
	//define struct to pair eigena value to its vector
	eigenPairs := make([]struct {
		value  float64
		vector []float64
	}, rows)
	for i := range values {
		eigenPairs[i].value = values[i]
		eigenPairs[i].vector = vectors[i]
	}
	//sorts in ascending ( greates to least)
	//worting eigen values results in sorting the vectors
	sort.SliceStable(eigenPairs, func(i, j int) bool {
		return eigenPairs[i].value > eigenPairs[j].value
	})
	for i := range values {
		values[i] = eigenPairs[i].value
		vectors[i] = eigenPairs[i].vector
	}

	return values, vectors
}

// Helper function to check difference between two vectors
func vectorDiff(v1, v2 []float64) float64 {
	diff := 0.0
	for i := range v1 {
		diff += (v1[i] - v2[i]) * (v1[i] - v2[i])
	}
	return math.Sqrt(diff)
}

func normalizeVector(vector []float64) []float64 {
	//magnitud formula using dot product
	magnitude := math.Sqrt(dotProduct(vector, vector))
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

func transposeMatrix(matrix [][]float64) [][]float64 {
	numRows := len(matrix)
	if numRows == 0 {
		return [][]float64{}
	}
	numCols := len(matrix[0])

	transposed := make([][]float64, numCols)
	for i := range transposed {
		transposed[i] = make([]float64, numRows)
		//change columns to rows
		for j := range transposed[i] {
			transposed[i][j] = matrix[j][i]
		}
	}
	return transposed
}
func calculatePCMatrix(normalizedMatrix [][]float64, eigenMatrix [][]float64) [][]float64 {
	// Obtains the length of the normalizedMatrix
	numRows := len(normalizedMatrix)

	// Obtains the length of the eigenMatrix
	numCols := len(eigenMatrix[0])

	pcMatrix := make([][]float64, numRows)
	for i := 0; i < numRows; i++ {
		// Reserve the memory for the new matrix
		pcMatrix[i] = make([]float64, numCols)

		for j := 0; j < numCols; j++ {
			sum := 0.0
			// Calculates the summation using the normalizedMatrix and the eigenMatrix
			for k := range normalizedMatrix[i] {
				sum += normalizedMatrix[i][k] * eigenMatrix[k][j]
			}

			// Assigns the calculated value to the item of the matrix
			pcMatrix[i][j] = sum
		}
	}

	// Returns the created matrix
	return pcMatrix
}

func calculateIndividualMatrix(principalMatrix [][]float64, normalizedMatrix [][]float64) [][]float64 {
	numRows := len(normalizedMatrix)
	numCols := len(normalizedMatrix[0])

	individualMatrix := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		denominator := 0.0
		// denominator for the row
		for k := 0; k < numCols; k++ {
			denominator += math.Pow(normalizedMatrix[i][k], 2)
		}

		individualMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			// Calculate the squared value for the current entry in the principalMatrix
			matrixCvalue := math.Pow(principalMatrix[i][j], 2)

			// Assign value
			individualMatrix[i][j] = matrixCvalue / denominator
		}
	}

	return individualMatrix
}

func columnToArray(matrix [][]float64, j int) []float64 {
	//return column from matrix as an array  given an index
	numRows := len(matrix)
	array := make([]float64, numRows)
	//j is the index received, so it is "static"
	for i := 0; i < numRows; i++ {
		array[i] = matrix[i][j]
	}
	return array
}

func rowToArray(matrix [][]float64, i int) []float64 {
	//return row from matrix as an array  given an index
	numCols := len(matrix[i])
	//reserve memory for the new array
	array := make([]float64, numCols)
	//method from GO LANG that copies one entire row to an array
	copy(array, matrix[i])
	return array
}

func calculateCoordinatesMatrix(pcMatrix, eigenMatrix [][]float64) [][]float64 {
	numRows := len(pcMatrix)
	numCols := len(eigenMatrix[0])
	coordinatesMatrix := make([][]float64, numRows)

	for i := 0; i < numRows; i++ {
		coordinatesMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			//obtains the column j of the eigenMatrix using the function created previously
			eigenColumn := columnToArray(eigenMatrix, j)
			//calculates and asigns the dotproduct to the new matrix
			coordinatesMatrix[i][j] = dotProduct(pcMatrix[i], eigenColumn)
		}
	}
	//returns the coordinatesMatrix
	return coordinatesMatrix
}
func calculateEigenvaluesAndVectors(pcMatrix [][]float64) ([]float64, [][]float64) {
	correlationPCmatrix := calculateCorrelationMatrix(pcMatrix)
	// Obtains a list of eigenvalues and eigenvectors just calling an existing function
	values, eigenMatrix := calculateEigenvaluesAndEigenvectors(correlationPCmatrix)
	// Receives the coordinatesMatrix
	coordinatesMatrix := calculateCoordinatesMatrix(pcMatrix, eigenMatrix)
	// Returns the eigenvalues and the CoordinatesMatrix
	return values, coordinatesMatrix
}

func calculateQualityV(vectores, coordinatesMatrix [][]float64) [][]float64 {

	// Inicializar matriz S
	var S = make([][]float64, len(vectores))
	for i := range S {
		S[i] = make([]float64, len(vectores))
	}

	// Multiplicar T * V^T
	for i := 0; i < len(S); i++ {
		for j := 0; j < len(S); j++ {
			for k := 0; k < len(vectores); k++ {
				S[i][j] += coordinatesMatrix[i][k] * vectores[j][k]
			}
		}
	}

	return S
}
func calculateInertiaVector(valoresPropios []float64) []float64 {

	// Obtener longitud de vectores
	m := float64(len(valoresPropios))

	// Inicializar vector de inercias
	inerciaVector := make([]float64, len(valoresPropios))

	// Calcular inercias
	for i, valorPropio := range valoresPropios {
		inerciaVector[i] = (valorPropio / m) * 100
	}

	return inerciaVector

}
