import axios from "axios"

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: process.env.VUE_APP_API_BASE_URL || "http://localhost:3000",
  timeout: 30000, // 30 seconds default timeout
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem("authToken")
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    // Add request timestamp for debugging
    config.metadata = { startTime: new Date() }

    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)

    return config
  },
  (error) => {
    console.error("❌ Request Error:", error)
    return Promise.reject(error)
  },
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    // Calculate request duration
    const duration = new Date() - response.config.metadata.startTime
    console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`)

    return response
  },
  (error) => {
    // Calculate request duration if available
    if (error.config?.metadata?.startTime) {
      const duration = new Date() - error.config.metadata.startTime
      console.error(`❌ API Error: ${error.config.method?.toUpperCase()} ${error.config.url} (${duration}ms)`)
    }

    // Handle common error scenarios
    if (error.response?.status === 401) {
      // Unauthorized - clear auth token and redirect to login
      localStorage.removeItem("authToken")
      // You could emit an event here to redirect to login page
    }

    if (error.response?.status >= 500) {
      console.error("Server Error:", error.response.data)
    }

    return Promise.reject(error)
  },
)

export default apiClient
