package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// preprocess preprocesses the text by tokenizing, converting to lowercase,
// and removing punctuation.
func preprocess(text string) []string {
	// Tokenization (split by space)
	tokens := strings.Fields(text)
	// Convert to lowercase
	for i, token := range tokens {
		tokens[i] = strings.ToLower(token)
	}
	// Remove punctuation
	var processedTokens []string
	for _, token := range tokens {
		token = strings.Trim(token, `,.!?"'`)
		processedTokens = append(processedTokens, token)
	}
	return processedTokens
}

// loadDataset loads the dataset from a CSV file.
func loadDataset(filename string) ([][]string, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data [][]string
	var labels []float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		data = append(data, record) // Fixed
		label := 0.0
		if record[1] == "positive" {
			label = 1.0
		}
		labels = append(labels, label)
	}
	return data, labels, nil
}

// createVocabulary creates a vocabulary from the dataset.
func createVocabulary(data [][]string) map[string]int {
	vocabulary := make(map[string]int)
	for _, review := range data {
		for _, token := range review {
			vocabulary[token]++
		}
	}
	return vocabulary
}

// createBagOfWords creates a bag-of-words representation of the dataset.
func createBagOfWords(data [][]string, vocabulary map[string]int) *mat.Dense {
	n := len(data)
	m := len(vocabulary)
	matrix := mat.NewDense(n, m, nil)

	for i, review := range data {
		for _, token := range review {
			if j, ok := vocabulary[token]; ok {
				matrix.Set(i, j, matrix.At(i, j)+1)
			}
		}
	}
	return matrix
}

// trainModel trains a logistic regression model using the bag-of-words
// representation and labels.
func trainModel(X *mat.Dense, y []float64) *mat.Dense {
	nSamples, nFeatures := X.Dims()
	Xb := mat.NewDense(nSamples, nFeatures+1, nil)
	for i := 0; i < nSamples; i++ {
		Xb.Set(i, 0, 1)
		for j := 0; j < nFeatures; j++ {
			Xb.Set(i, j+1, X.At(i, j))
		}
	}

	theta := mat.NewDense(nFeatures+1, 1, nil)

	alpha := 0.01 // learning rate
	iterations := 1000
	for iter := 0; iter < iterations; iter++ {
		h := mat.NewDense(nSamples, 1, nil)
		h.Mul(Xb, theta)
		h.Apply(func(i, j int, v float64) float64 { return sigmoid(v, 0) }, h)

		errors := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			errors[i] = y[i] - h.At(i, 0)
		}

		gradient := mat.NewDense(nFeatures+1, 1, nil)
		gradient.Mul(Xb.T(), mat.NewDense(nSamples, 1, errors))
		gradient.Scale(alpha/float64(nSamples), gradient)

		theta.Add(theta, gradient)

		// Print progress
		fmt.Printf("Iteration %d/%d\n", iter+1, iterations)
	}
	return theta
}

// sigmoid computes the sigmoid function.
func sigmoid(x, _ float64) float64 {
	return 1 / (1 + math.Exp(-x)) // Fixed
}

// predict predicts the sentiment of a review using the trained model.
func predict(review string, vocabulary map[string]int, theta *mat.Dense) float64 {
	tokens := preprocess(review)
	X := createBagOfWords([][]string{tokens}, vocabulary)
	nSamples, _ := X.Dims()
	Xb := mat.NewDense(nSamples, X.RawMatrix().Cols+1, nil)
	for i := 0; i < nSamples; i++ {
		Xb.Set(i, 0, 1)
		for j := 0; j < X.RawMatrix().Cols; j++ {
			Xb.Set(i, j+1, X.At(i, j))
		}
	}

	h := mat.NewDense(nSamples, 1, nil)
	h.Mul(Xb, theta)
	h.Apply(func(i, j int, v float64) float64 { return sigmoid(v, 0) }, h) // Fixed
	return h.At(0, 0)
}

func main() {
	// Load dataset
	data, labels, err := loadDataset("imdbdataset.csv")
	if err != nil {
		fmt.Println("Error loading dataset:", err)
		return
	}

	// Create vocabulary
	vocabulary := createVocabulary(data)

	// Create bag-of-words representation
	X := createBagOfWords(data, vocabulary)

	// Train model
	theta := trainModel(X, labels)

	// Example prediction
	review := "This movie was great!"
	sentiment := predict(review, vocabulary, theta)
	if sentiment > 0.5 {
		fmt.Println("Positive sentiment")
	} else {
		fmt.Println("Negative sentiment")
	}
}
