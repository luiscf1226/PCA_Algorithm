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

	//calculate the mean and standard deviation
	//each method will return a matrix
	means := calculateColumnMeans(matrix)
	stdDevs := calculateColumnStdDevs(matrix)

	normalizedMatrix := normalizeMatrix(matrix, means, stdDevs)
	printMatrix(normalizedMatrix)
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

// Imprime una matriz
func printMatrix(matrix [][]float64) {
	for _, row := range matrix {
		fmt.Println(row)
	}
}
