package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

const (
	matrixSize = 5
	precision  = 1e-6
	maxIter    = 1000
)

func main() {

	filePath := "EjemploEstudiantes.csv"
	//from file path read file or return error
	matrix, err := readCSVFile(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	print("Matrix Received by file \n")
	printMatrix(matrix)
	//Implement first step 1 center and reduce normalized matrix
	//obtain matrix by calculating mean and deviation standard to normalize
	means := calculateColumnMeans(matrix)
	stdDevs := calculateColumnStdDevs(matrix)
	normalizedMatrix := normalizeMatrix(matrix, means, stdDevs)
	print("Matrix Normalized \n")
	printMatrix(normalizedMatrix)

	//Second step 2 Correlation Matrix
	correlationMatrix := calculateCorrelationMatrix(normalizedMatrix)
	print("Correlation Matrix \n")
	printMatrix(correlationMatrix)

	//third step order by greatest to least vectors
	//obtain list of proper values and proper vectors
	values, vectors := calculateEigenvaluesAndEigenvectors(correlationMatrix)

	//print eigen and vector values
	fmt.Println("Eigenvalues:", values)
	fmt.Println("Eigenvectors:")
	for _, vector := range vectors {
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
	//calculate the total by adding each value inside matrix
	for _, row := range matrix {
		for i, val := range row {
			means[i] += val
		}
	}
	//we then get means value for each column and devide by the total to get average
	for i := range means {
		means[i] /= float64(len(matrix))
	}
	return means
}

func calculateColumnStdDevs(matrix [][]float64) []float64 {
	//calculate column means again to get ready for standard deviation
	means := calculateColumnMeans(matrix)
	stdDevs := make([]float64, len(matrix[0]))
	for _, row := range matrix {
		for i, val := range row {
			//calculate standard deviation used formula
			//dv=squar root of actual (value-means)^2
			stdDevs[i] += math.Pow(val-means[i], 2)
		}
	}
	for i := range stdDevs {
		//we then use the rest of formula
		//dv=(method above)/total number of elements in matrix
		stdDevs[i] = math.Sqrt(stdDevs[i] / float64(len(matrix)))
	}
	return stdDevs
}

// Normalize using z score
// this ensure data is represented mean between 0 and standard deviation 1
func normalizeMatrix(matrix [][]float64, means, stdDevs []float64) [][]float64 {
	for i, row := range matrix {
		for j, val := range row {
			//formula for z score
			//actual index=(value-mean) divided by value of standard deviation
			matrix[i][j] = (val - means[j]) / stdDevs[j]
		}
	}
	return matrix
}

// Print Matrix function
func printMatrix(matrix [][]float64) {
	for _, row := range matrix {
		fmt.Println(row)
	}
}

func calculateCorrelationMatrix(normalizedMatrix [][]float64) [][]float64 {
	//get number of columns and initalize matrix
	numCols := len(normalizedMatrix[0])
	correlationMatrix := make([][]float64, numCols)
	//get means from function
	means := calculateColumnMeans(normalizedMatrix)

	for i := 0; i < numCols; i++ {
		correlationMatrix[i] = make([]float64, numCols)
		for j := 0; j < numCols; j++ {
			//for each value get correlation value for each matrix
			correlation := calculateCorrelationValue(normalizedMatrix[i], normalizedMatrix[j], means[i], means[j])
			correlationMatrix[i][j] = correlation
		}
	}
	return correlationMatrix
}

func calculateCorrelationValue(col1, col2 []float64, mean1, mean2 float64) float64 {
	//values for data points and formula
	n := len(col1)
	numerator := 0.0
	denominator1 := 0.0
	denominator2 := 0.0

	for i := 0; i < n; i++ {
		//get numerator by getting val1,val2 and substracting each by mean
		numerator += (col1[i] - mean1) * (col2[i] - mean2)
		//denominator is obtained by getting the value - mean to power of 2
		denominator1 += math.Pow(col1[i]-mean1, 2)
		denominator2 += math.Pow(col2[i]-mean2, 2)
	}
	//return correlation value with formula
	correlation := numerator / (math.Sqrt(denominator1) * math.Sqrt(denominator2))
	return correlation
}
func calculateEigenvaluesAndEigenvectors(R [][]float64) ([]float64, [][]float64) {

	rows := len(R)
	values := make([]float64, rows)
	vectors := make([][]float64, rows)
	// get random vector and normalize to get the values
	for i := 0; i < rows; i++ {
		vector := make([]float64, rows)
		for j := 0; j < rows; j++ {
			vector[j] = rand.Float64()
		}
		//send random vector from matrix to normalize
		vector = normalizeVector(vector)

		for iter := 0; iter < maxIter; iter++ {
			//obtain next vector by multiplying an normalizing
			nextVector := multiplyMatrixVector(R, vector)
			nextVector = normalizeVector(nextVector)
			//eigen value from dotproduct
			eigenvalue := dotProduct(nextVector, vector)
			//obtain absoult value if its less than precision
			if math.Abs(eigenvalue-values[i]) < precision {
				break
			}

			vector = nextVector
		}
		//retuirn values from dotProduct function and vectors
		values[i] = dotProduct(multiplyMatrixVector(R, vector), vector)
		vectors[i] = vector
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
