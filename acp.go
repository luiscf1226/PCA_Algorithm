package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
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
		for _, field := range fields[1:] { // Omitir el primer campo (nombre)
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
