package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

func init() {
	if err := godotenv.Load(); err != nil {
		fmt.Println("No .env file found")
	}
}

var tmpl = template.Must(template.ParseGlob("templates/*.html"))

type ResultData struct {
	Prompt   string
	Response string
	PastConversations []Message
}

type OpenAIRequest struct {
	Model   string        `json:"model"`
	Messages []Message     `json:"messages"`
	MaxTokens int          `json:"max_tokens"`
	Temperature float64    `json:"temperature"`
	TopP float64           `json:"top_p"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty float64  `json:"presence_penalty"`
	Stop []string           `json:"stop"`
}

type Message struct {
	Role string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponse struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Created int `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Message Message     `json:"message"`
	} `json:"choices"`
}

func main() {
	http.HandleFunc("/", indexHandler)
	http.HandleFunc("/submit", submitHandler)
	http.ListenAndServe(":8080", nil)
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	tmpl.ExecuteTemplate(w, "index.html", nil)
}

func submitHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request body
	var data ResultData
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	prompt := data.Prompt
	if prompt == "" {
		http.Error(w, "Prompt is required", http.StatusBadRequest)
		return
	}

	apiKey, ok := os.LookupEnv("OPENAI_API_KEY")
	if !ok {
		http.Error(w, "API key not found in environment variables", http.StatusInternalServerError)
		return
	}

	// Call GPT-3 API
  pastConversations := data.PastConversations
  response, err := callGPT3API(prompt, apiKey, pastConversations)

	if err != nil {
		fmt.Println("submitHandler error:", err)

		http.Error(w, "Error calling GPT-3 API", http.StatusInternalServerError)
		return
	}

	data.Response = response

	// Set header for JSON response
	w.Header().Set("Content-Type", "application/json")

	// Send response
	json.NewEncoder(w).Encode(data)
}

func callGPT3API(prompt, apiKey string, pastConversations []Message) (string, error) {
	apiURL := "https://api.openai.com/v1/chat/completions"

	messages := append([]Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
	}, pastConversations...)

	messages = append(messages, Message{
		Role:    "user",
		Content: prompt,
	})

	requestBody := OpenAIRequest{
		Model:            "gpt-3.5-turbo",
		Messages:          messages,
		MaxTokens:         4000,
		Temperature:       1,
		TopP:              1,
		FrequencyPenalty:  0.0,
		PresencePenalty:   0.6,
		Stop:              []string{" Human:", " AI:"},
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	client := &http.Client{}
	req, err := http.NewRequest("POST", apiURL, strings.NewReader(string(body)))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := ioutil.ReadAll(resp.Body)
		return "", errors.New(string(bodyBytes))
	}

	var apiResponse OpenAIResponse
	err = json.NewDecoder(resp.Body).Decode(&apiResponse)
	if err != nil {
		return "", err
	}

	if len(apiResponse.Choices) == 0 {
		return "", errors.New("No choices returned by API")
	}

	return strings.TrimSpace(apiResponse.Choices[0].Message.Content), nil
}
