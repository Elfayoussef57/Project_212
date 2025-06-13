import axios from "axios"

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: 'http://localhost:5000', // ← à adapter si ton backend tourne ailleurs
  timeout: 300000,
})

export default apiClient
